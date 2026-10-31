import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLLMVTCCompressor

class PyramidDrop(BaseLLMVTCCompressor):
    """Attention-based visual-token selection for NEO's Qwen3 decoder."""

    def __init__(self, compression_ratio: float = 0.5, compression_layer: int = 2):
        super().__init__(compression_ratio=compression_ratio, compression_layer=compression_layer)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """
        x:   [B, H, L, D]
        cos: [B, L, D]
        sin: [B, L, D]
        """
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        return x * cos + rotated * sin

    @torch.no_grad()
    def select_indices(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        visual_mask: torch.Tensor,
        next_layer: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [1, L, hidden_size]
            indexes:       [3, L] (temporal, height, width)
            visual_mask:   [L], True for visual tokens
            next_layer:    Decoder layer immediately after the pruning point

        Returns:
            keep_indices: [L_new], sorted original token positions
        """
        B, L, _ = hidden_states.shape

        if B != 1:
            raise NotImplementedError("PyramidDrop currently supports batch size 1.")

        if indexes.shape != (3, L):
            raise ValueError(f"Expected indexes [3, {L}], got {indexes.shape}")

        visual_mask = visual_mask.to(device=hidden_states.device, dtype=torch.bool).reshape(-1)

        if visual_mask.numel() != L:
            raise ValueError("visual_mask must have one entry per token.")

        visual_indices = torch.where(visual_mask)[0]
        num_visual = visual_indices.numel()

        if num_visual == 0 or self.compression_ratio == 1.0:
            return torch.arange(L, device=hidden_states.device)

        k = max(1, min(round(num_visual * self.compression_ratio), num_visual))
        attn = next_layer.self_attn

        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim

        # 1. Compute Q for the final prompt token and K for all tokens
        query_hidden = hidden_states[:, -1:, :]

        q_shape = (B, 1, num_heads, head_dim)
        k_shape = (B, L, num_kv_heads, head_dim)

        q_t = attn.q_norm(attn.q_proj(query_hidden).view(q_shape)).transpose(1, 2)

        k_t = attn.k_norm(attn.k_proj(hidden_states).view(k_shape)).transpose(1, 2)

        # NEO's spatial projections produce two components.
        q_h, q_w = attn.q_proj_hw(query_hidden).view(q_shape).transpose(1, 2).chunk(2, dim=-1)
        k_h, k_w = attn.k_proj_hw(hidden_states).view(k_shape).transpose(1, 2).chunk(2, dim=-1)
        q_h = attn.q_norm_h(q_h)
        q_w = attn.q_norm_w(q_w)
        k_h = attn.k_norm_h(k_h)
        k_w = attn.k_norm_w(k_w)

        # 2. Apply temporal and spatial rotary embeddings
        q_indexes = indexes[:, -1:]
        k_indexes = indexes

        cos_q_t, sin_q_t = attn.rotary_emb(query_hidden, q_indexes[0].unsqueeze(0))
        cos_k_t, sin_k_t = attn.rotary_emb(hidden_states, k_indexes[0].unsqueeze(0))
        cos_q_h, sin_q_h = attn.rotary_emb_hw(query_hidden, q_indexes[1].unsqueeze(0))
        cos_k_h, sin_k_h = attn.rotary_emb_hw(hidden_states, k_indexes[1].unsqueeze(0))
        cos_q_w, sin_q_w = attn.rotary_emb_hw(query_hidden, q_indexes[2].unsqueeze(0))
        cos_k_w, sin_k_w = attn.rotary_emb_hw(hidden_states, k_indexes[2].unsqueeze(0))

        q_t = self._apply_rope(q_t, cos_q_t, sin_q_t)
        k_t = self._apply_rope(k_t, cos_k_t, sin_k_t)
        q_h = self._apply_rope(q_h, cos_q_h, sin_q_h)
        k_h = self._apply_rope(k_h, cos_k_h, sin_k_h)
        q_w = self._apply_rope(q_w, cos_q_w, sin_q_w)
        k_w = self._apply_rope(k_w, cos_k_w, sin_k_w)

        q = torch.cat([q_t, q_h, q_w], dim=-1)
        key = torch.cat([k_t, k_h, k_w], dim=-1)

        # 3. Grouped-query attention
        num_groups = num_heads // num_kv_heads
        key = torch.repeat_interleave(key, num_groups, dim=1)
        # q:   [1, num_heads, 1, D_total]
        # key: [1, num_heads, L, D_total]
        scores = torch.matmul(q, key.transpose(-2, -1)) * attn.scaling
        attention = F.softmax(scores.float(), dim=-1)
        importance = attention.mean(dim=1).squeeze(0).squeeze(0)

        # 4. Select top-K visual tokens
        visual_scores = importance[visual_indices]
        topk = visual_scores.topk(k=k, dim=-1).indices
        selected_visual = visual_indices[topk]

        # Always preserve nonvisual tokens.
        keep_mask = ~visual_mask
        keep_mask[selected_visual] = True
        keep_indices = torch.where(keep_mask)[0]
        return keep_indices
