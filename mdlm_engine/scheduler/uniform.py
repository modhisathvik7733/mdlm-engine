"""Uniform scheduler — equal tokens-per-step.

Generalizes Dream-Coder's ``get_num_transfer_tokens`` from
``fast_dllm/generation_utils_block.py:37-60``: at step ``t`` of
``steps_per_block``, commit the top-K most confident positions where
``K = ceil(remaining_masked / (steps_per_block - t))``.

Simple, deterministic, no threshold — useful as the equivalence baseline.
"""
from __future__ import annotations

import math

import torch

from mdlm_engine.scheduler.base import register_scheduler


@register_scheduler("uniform")
def uniform_scheduler(
    confidences: torch.Tensor,    # [N]
    mask_index: torch.Tensor,     # [B, block_len], True at masked positions
    step: int,
    steps_per_block: int,
    threshold: float = 0.9,        # unused
) -> torch.Tensor:
    """Commit ceil(remaining/steps_left) most-confident positions per sample."""
    del threshold  # uniform doesn't use it
    if confidences.numel() == 0:
        return torch.zeros_like(confidences, dtype=torch.bool)

    B = mask_index.shape[0]
    steps_left = max(1, steps_per_block - step)
    commit_mask = torch.zeros_like(confidences, dtype=torch.bool)

    # confidences are flat [N]; we need to map them back to per-sample groups.
    # The engine guarantees confidences are emitted in the same order as the
    # masked positions: row-major over (sample, position) for masked entries.
    counts = mask_index.sum(dim=-1)  # [B] — masked count per sample
    starts = torch.zeros(B + 1, dtype=torch.long, device=confidences.device)
    torch.cumsum(counts, dim=0, out=starts[1:])

    for b in range(B):
        s, e = int(starts[b]), int(starts[b + 1])
        n_masked = e - s
        if n_masked == 0:
            continue
        n_commit = min(n_masked, math.ceil(n_masked / steps_left))
        if n_commit == 0:
            continue
        sample_conf = confidences[s:e]
        # Top-n_commit indices (within this sample's slice).
        topk_idx = torch.topk(sample_conf, n_commit).indices
        commit_mask[s:e][topk_idx] = True

    return commit_mask
