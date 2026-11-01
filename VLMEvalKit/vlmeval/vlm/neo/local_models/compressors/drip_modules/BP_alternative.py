import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def box(x):
    return F.relu(x) - F.relu(x - 1)

def diff_downsample(boundaries: torch.Tensor, hidden: torch.Tensor, null_group: torch.Tensor):
    B, L = boundaries.shape
    _, _, D = hidden.shape

    # Keep boundaries differentiable!
    seg_counts = (boundaries.detach() > 0.5).sum(dim=1)
    S = int(seg_counts.max().item())

    if S == 0:
        return null_group.expand(1, B, D).to(device=hidden.device, dtype=hidden.dtype)

    seg_ids = torch.arange(S, device=boundaries.device, dtype=boundaries.dtype,).view(1, 1, S)

    # Regular cumsum — no detached_cumsum.
    cumulative = boundaries.cumsum(dim=1)
    real_segment_index = cumulative - boundaries

    # Differentiable replacement for equality comparison.
    diff = real_segment_index.unsqueeze(-1) - seg_ids
    membership = box(diff + 1) - box(diff)

    # Keep normalization differentiable.
    denom = membership.sum(dim=1, keepdim=True).clamp_min(1e-9)
    weights = membership / denom

    shortened_hidden = torch.einsum("lbd,bls->sbd", hidden, weights.to(hidden.dtype))
    return shortened_hidden