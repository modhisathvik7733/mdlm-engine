"""Confidence-threshold scheduler — port of fast_dllm's threshold logic.

Original at ``fast_dllm/generation_utils_block.py:574-637``: commit any
position whose confidence exceeds ``threshold``. If no position passes
the threshold, commit the single most-confident position (so progress is
guaranteed each step). Useful as a "let the model self-pace" scheduler.
"""
from __future__ import annotations

import torch

from mdlm_engine.scheduler.base import register_scheduler


@register_scheduler("confidence")
def confidence_scheduler(
    confidences: torch.Tensor,    # [N]
    mask_index: torch.Tensor,     # [B, block_len]
    step: int,
    steps_per_block: int,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Commit positions whose confidence exceeds ``threshold``.

    Per-sample tie-breaker: if NO position in a sample passes the threshold,
    commit that sample's single most-confident position so the diffusion
    loop always makes forward progress.
    """
    del step, steps_per_block  # this scheduler is step-independent
    if confidences.numel() == 0:
        return torch.zeros_like(confidences, dtype=torch.bool)

    B = mask_index.shape[0]
    counts = mask_index.sum(dim=-1)
    starts = torch.zeros(B + 1, dtype=torch.long, device=confidences.device)
    torch.cumsum(counts, dim=0, out=starts[1:])

    commit_mask = confidences >= threshold

    # Per-sample fallback: if no commits in a sample, commit the argmax.
    for b in range(B):
        s, e = int(starts[b]), int(starts[b + 1])
        if e == s:
            continue
        if not commit_mask[s:e].any():
            argmax_local = int(confidences[s:e].argmax())
            commit_mask[s + argmax_local] = True

    return commit_mask
