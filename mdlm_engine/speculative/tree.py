"""Tree speculative decoding (v0.4.0) — position-band branches over SSD.

v0.3.0's single-branch SSD at threshold=0.99 commits all positions whose
top-1 softmax probability ≥ 0.99 in one verify forward. v0.4.0 adds a
SECOND verify forward (branch 1) that targets positions in the band
``[band_low, 0.99)`` — positions the model is "almost" 99% confident
about, which after branch 0's commits get written into state.x may rise
above 0.99 due to richer committed context.

Branch 1 is committed only if its argmax under branch-0's post-commit
context still matches the proposed token. By construction this is
lossless: argmax-match at top_p=0.95 sampling implies the regular
sampler would have committed the same token at that position.

Branches are DISJOINT by confidence-band partition:
- Branch 0: positions whose softmax-max is ≥ 0.99
- Branch 1: positions whose softmax-max is in ``[band_low, 0.99)``

Branch 1's positions are NOT proposed in branch 0 (strict ``< 0.99``).
This guarantees branch 1 finds NEW commits, never duplicates branch 0.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mdlm_engine.speculative.propose import (
    Proposal, _empty_proposal, propose_block_level,
)

if TYPE_CHECKING:
    import torch


def propose_band(
    block_logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    block_start: int,
    *,
    band_low: float,
    band_high: float,
    max_proposals: int = 4,
) -> Proposal:
    """Pick masked positions whose top-1 softmax probability is in
    ``[band_low, band_high)``. Used for tree-spec branch 1+.

    Differs from ``propose_block_level``:
    - Two thresholds (low and high) defining a band, not just a floor.
    - Strict upper bound (< band_high) so branch 1's positions are
      DISJOINT from branch 0's (which uses confidence_threshold=band_high).
    - max_proposals cap is mandatory (default 4) to bound verify-forward
      cost when the band is wide and many positions clear it.

    Parameters
    ----------
    block_logits : ``[B=1, block_len, V]``
        Active-block logits, already shift_logits'd by the adapter.
    mask_index : ``[B=1, block_len]`` bool
        True at currently-masked active-block positions.
    block_start : int
        Absolute index of the first active-block position in state.x.
    band_low : float
        Inclusive lower bound on top-1 softmax probability.
    band_high : float
        EXCLUSIVE upper bound. Must be > band_low.
    max_proposals : int
        Hard cap on result count after band filter (default 4).

    Returns
    -------
    Proposal
        Up to ``max_proposals`` positions in the band, sorted by
        descending confidence. Empty if no positions fall in the band
        or if mask_index has fewer than 2 masked positions
        (leave-one-masked rule).

    Raises
    ------
    ValueError
        If ``band_low >= band_high`` or ``band_low <= 0`` or ``band_high > 1``.
    """
    import torch

    if band_low >= band_high:
        raise ValueError(
            f"propose_band requires band_low < band_high; "
            f"got band_low={band_low}, band_high={band_high}."
        )
    if band_low <= 0 or band_high > 1.0:
        raise ValueError(
            f"propose_band requires 0 < band_low < band_high <= 1.0; "
            f"got [{band_low}, {band_high})."
        )
    if max_proposals <= 0:
        return _empty_proposal(block_logits.device)

    B, block_len, V = block_logits.shape
    if B != 1:
        # Multi-sample tree spec deferred. v0.4.0 ships single-sample only.
        return _empty_proposal(block_logits.device)

    proposable = mask_index[0]  # [block_len] bool
    n_proposable = int(proposable.sum())
    if n_proposable <= 1:
        # Leave-one-masked rule: need at least 2 masked positions.
        return _empty_proposal(block_logits.device)

    proposable_block_indices = proposable.nonzero(as_tuple=True)[0]  # [n_proposable]
    proposable_logits = block_logits[0, proposable_block_indices, :]  # [n_proposable, V]
    probs = torch.softmax(proposable_logits, dim=-1)
    top_p, top_tok = probs.max(dim=-1)  # [n_proposable]

    # Band filter: [band_low, band_high) — strict upper bound.
    in_band = (top_p >= band_low) & (top_p < band_high)
    if not bool(in_band.any()):
        return _empty_proposal(block_logits.device)

    selected_block_idx = proposable_block_indices[in_band]
    selected_tokens = top_tok[in_band]
    selected_confidences = top_p[in_band]

    # Sort descending by confidence.
    order = torch.argsort(selected_confidences, descending=True)
    selected_block_idx = selected_block_idx[order]
    selected_tokens = selected_tokens[order]
    selected_confidences = selected_confidences[order]

    # Hard cap.
    if selected_block_idx.numel() > max_proposals:
        selected_block_idx = selected_block_idx[:max_proposals]
        selected_tokens = selected_tokens[:max_proposals]
        selected_confidences = selected_confidences[:max_proposals]

    selected_positions = selected_block_idx + block_start
    return Proposal(
        positions=selected_positions.to(torch.long),
        tokens=selected_tokens.to(torch.long),
        confidences=selected_confidences.float(),
    )


def propose_tree(
    block_logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    block_start: int,
    *,
    high_threshold: float = 0.99,
    band_low: float = 0.97,
    max_proposals_branch_1: int = 4,
) -> "tuple[Proposal, Proposal]":
    """Build a 2-branch tree-spec proposal.

    Branch 0: positions ≥ ``high_threshold`` (today's v0.3.0 default).
    Branch 1: positions in ``[band_low, high_threshold)`` — disjoint
    from branch 0 by construction.

    Returns
    -------
    (branch_0, branch_1) : tuple[Proposal, Proposal]
        Both proposals are disjoint and may be empty.

    Notes
    -----
    Branch 0's caller should run verify FIRST and commit accepted
    positions. Branch 1's verify must run with branch-0's commits
    visible in state.x (caller's responsibility — we don't mutate
    state here).
    """
    branch_0 = propose_block_level(
        block_logits=block_logits,
        mask_index=mask_index,
        block_start=block_start,
        confidence_threshold=high_threshold,
    )
    branch_1 = propose_band(
        block_logits=block_logits,
        mask_index=mask_index,
        block_start=block_start,
        band_low=band_low,
        band_high=high_threshold,
        max_proposals=max_proposals_branch_1,
    )
    return branch_0, branch_1


__all__ = ["propose_band", "propose_tree"]
