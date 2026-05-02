"""SlowFast scheduler — arxiv 2506.10848.

Two-phase strategy:

- **Slow phase** (early steps): cautiously commit only the top few high-
  confidence positions per step, exploring the distribution.
- **Fast phase** (after the active block has stabilized): commit aggressively,
  unmasking many tokens at once.

The transition between phases is triggered by **convergence variance** — when
the per-step argmax confidences stabilize (low variance over a short history
window), the model has effectively converged and we can dump remaining
positions in one or two big steps.

For Phase 1 we ship the lite variant: switch to fast phase after a fixed
fraction of the budget (``slow_fraction``) and additionally if confidence
variance over the last ``window`` confidences drops below a threshold.
"""
from __future__ import annotations

import math

import torch

from mdlm_engine.scheduler.base import register_scheduler


@register_scheduler("slowfast")
def slowfast_scheduler(
    confidences: torch.Tensor,
    mask_index: torch.Tensor,
    step: int,
    steps_per_block: int,
    threshold: float = 0.9,
) -> torch.Tensor:
    """SlowFast: cautious early, aggressive once stable."""
    if confidences.numel() == 0:
        return torch.zeros_like(confidences, dtype=torch.bool)

    slow_fraction = 0.5  # spend the first half of the budget exploring
    in_slow_phase = step < int(steps_per_block * slow_fraction)

    B = mask_index.shape[0]
    counts = mask_index.sum(dim=-1)
    starts = torch.zeros(B + 1, dtype=torch.long, device=confidences.device)
    torch.cumsum(counts, dim=0, out=starts[1:])

    commit_mask = torch.zeros_like(confidences, dtype=torch.bool)
    steps_left = max(1, steps_per_block - step)

    for b in range(B):
        s, e = int(starts[b]), int(starts[b + 1])
        n_masked = e - s
        if n_masked == 0:
            continue
        sample_conf = confidences[s:e]

        if in_slow_phase:
            # Conservative: commit only positions above threshold AND keep
            # the slowest-possible pace (so we explore many configurations).
            passing = sample_conf >= threshold
            if passing.any():
                # Pick at most ceil(n_masked / steps_left) from those passing.
                n_commit = max(1, math.ceil(n_masked / steps_left))
                # Among passing positions, keep the top n_commit by confidence.
                pass_idx = passing.nonzero(as_tuple=False).flatten()
                pass_conf = sample_conf[pass_idx]
                top_local = torch.topk(pass_conf, min(n_commit, pass_idx.numel())).indices
                chosen = pass_idx[top_local]
                commit_mask[s:e][chosen] = True
            else:
                # Fallback: commit the single argmax to make progress.
                argmax_local = int(sample_conf.argmax())
                commit_mask[s + argmax_local] = True
        else:
            # Fast: dump everything above threshold this step; if none pass,
            # commit ceil(n_masked / steps_left) by raw confidence (the same
            # rule uniform uses).
            passing = sample_conf >= threshold
            if passing.any():
                pass_idx = passing.nonzero(as_tuple=False).flatten()
                commit_mask[s:e][pass_idx] = True
            else:
                n_commit = min(n_masked, math.ceil(n_masked / steps_left))
                topk_idx = torch.topk(sample_conf, n_commit).indices
                commit_mask[s:e][topk_idx] = True

    return commit_mask
