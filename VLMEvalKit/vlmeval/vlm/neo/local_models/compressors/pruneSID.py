from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import math

from .base import BaseVTCCompressor
from .registry import register_compressor


#adapted from: https://github.com/ZhengyaoFang/PruneSID/blob/main/prunesid/prunesid_qwen/modeling_qwen2_vl.py
def pca_group(features, min_components=32):
    standard_features = torch.sigmoid(features.to(torch.float32)).permute(1,0)
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=1)
    return V, belong_components

def nms(similarity_matrix, scores, threshold):
    keep = []
    while scores.sum() > 0:
        max_idx = scores.argmax(axis=0)
        scores[max_idx] = 0
        keep.append(max_idx)
        condition = similarity_matrix[max_idx] > threshold
        scores[condition] = 0
    return keep

@torch.no_grad()
def prunesid_select(hidden_states: torch.Tensor, need_token_num: int) -> torch.Tensor:
    """hidden_states: [N, D] tokens of one image. Returns the sorted indices of the need_token_num kept tokens."""
    projector_lengths, belong_components = pca_group(hidden_states, min_components=min(max(int(need_token_num / 4), 4), *hidden_states.shape))
    projector_scores = projector_lengths.clone()

    projector_mask = belong_components.unsqueeze(1).repeat(1, projector_lengths.shape[1])
    index_map = torch.arange(projector_lengths.shape[1], device=projector_lengths.device).unsqueeze(0).repeat(projector_lengths.shape[0],1)
    weights_mask = torch.where(projector_mask != index_map)
    projector_lengths[weights_mask] = 0
    projector_scores[weights_mask] = 0

    normalized_states = F.normalize(hidden_states.float(), p=2, dim=-1)
    group_similarity = torch.bmm(normalized_states.unsqueeze(0), normalized_states.T.unsqueeze(0))[0] # [batch_size, 576, 576]
    sim_mean = group_similarity.triu(diagonal=0).mean()
    group_similarity = group_similarity.to(torch.float32).cpu().numpy()
    group_idxs = []

    
    ratio = max(need_token_num / ((hidden_states.shape[0]) / 18),1)
    given_scores = torch.arange(projector_lengths.shape[0],0,-1, device=projector_lengths.device, dtype=projector_lengths.dtype)
    for g in range(projector_lengths.shape[1]):
        group_indices = torch.where(belong_components == g)[0].cpu().numpy()
        if group_indices.shape[0] == 0:
            group_idxs.append(np.array([]))
            continue
        g_similarity = group_similarity[group_indices, :][ :, group_indices]
        g_scores = projector_lengths[:,g][group_indices].cpu().numpy()
        keep_indices = nms(g_similarity, g_scores, float(ratio * sim_mean))
        keep_indices = group_indices[keep_indices]
        projector_scores[keep_indices,g] = given_scores[:keep_indices.shape[0]]
        group_idxs.append(keep_indices)    
    
    keep_nms_counts = torch.tensor([group_idxs[i].shape[0] for i in range(len(group_idxs))], device=projector_lengths.device)
    group_counts = (projector_mask == index_map).sum(dim=0)


    group_lower_bound = torch.ones(group_counts.shape[0], device=group_counts.device)
    group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

    group_upper_bound = torch.ones(group_counts.shape[0], device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
    while group_upper_bound.sum() < need_token_num:
        group_upper_bound = group_upper_bound + 1
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0] 
    
    other_token_nums = max(0, need_token_num - group_lower_bound.sum())
    norm_group_counts = keep_nms_counts / keep_nms_counts.sum()
    cumulative_sum = torch.cumsum(norm_group_counts, dim=0)
    other_token_d = (cumulative_sum * other_token_nums).round().int()
    other_token_d = other_token_d - torch.cat([torch.zeros(1, device=other_token_d.device), other_token_d[:-1]])
    group_token_d = other_token_d + group_lower_bound


    group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
    group_mean_sort_index = torch.argsort(keep_nms_counts, descending=True, dim=0)
    filling_group = 0

    while group_token_d.sum() < other_token_nums + group_lower_bound.sum():
        filling_num = min(group_upper_bound[group_mean_sort_index[filling_group]] - group_token_d[group_mean_sort_index[filling_group]], other_token_nums + group_lower_bound.sum() - group_token_d.sum())
        group_token_d[group_mean_sort_index[filling_group]] += filling_num
        filling_group += 1
    # print(group_token_d.sum(), keep_nms_counts.sum())
    projector_sort_index = torch.argsort(projector_scores, descending=True, dim=0)
    important_indices = []
    for g in range(len(group_token_d)):
        important_indices.append(projector_sort_index[:,g][:int(group_token_d[g])])
    important_indices = torch.cat(important_indices, dim=0)[:need_token_num]
    # raster order for the LLM
    important_indices = important_indices.sort()[0]
    assert important_indices.numel() == need_token_num, (
        f"PruneSID kept {important_indices.numel()} tokens, expected {need_token_num}"
    )
    return important_indices


@register_compressor("prunesid")
class PruneSIDCompressor(BaseVTCCompressor):
    """PruneSID (Training Free) Token Pruning: https://arxiv.org/pdf/2603.09480
       very similar to official implementation, we just fix the compression ration rather than from computed tau
    """

    def __init__(self, compression_ratio: float = 1.0, seed: int = 0):
        super().__init__(compression_ratio)
        self.seed = seed

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError(
            "PruneSID chooses its tokens from the image features, not from (h, w): "
            "use forward_with_indices (training and precomputed visual_features are not supported)."
        )

    def num_output_tokens(self, h: int, w: int) -> int:
        return self._get_target_num_tokens(h * w)

    @torch.no_grad()
    def forward_with_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._flatten_tokens(x)  # [B, N, D]
        B, N, D = x.shape
        assert B == 1, "PruneSID compresses one image at a time"

        K = self._get_target_num_tokens(N)

        if K == N: return x, torch.arange(N, device=x.device)

        # pca_lowrank is randomized: seed it w/out touching the global RNG for determinism
        devices = [x.device] if x.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            idx = prunesid_select(x[0], K).to(x.device)
        return x[:, idx], idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_indices(x)[0]

