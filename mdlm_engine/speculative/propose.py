"""Self-speculative proposal step — pick K masked positions to speculate on.

Given the per-step logits the engine ALREADY computed (no extra forward),
identify the K masked positions with the HIGHEST sampling confidence —
those are the positions where the model is most certain, so a parallel
speculative commit is most likely to survive verification.

Why "highest confidence" not "lowest":
- The naive intuition "speculate at LOW confidence to fill in tricky ones"
  fails: low-confidence proposals usually get rejected by verification
  (acceptance rate < 50%).
- High-confidence proposals look "boring" but the engine would commit
  them anyway over the next few steps. SSD just commits them NOW in
  parallel, saving forwards.
- The arxiv 2510.04147 paper's draft heuristic is essentially "argmax
  + confidence sort" — see section 3.1 ("Drafting").

Returns the k positions + their argmax tokens; the verifier then runs ONE
forward with these positions filled in and checks the prediction agrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Proposal:
    """One round of speculation on the active block.

    Attributes
    ----------
    positions :
        ``[n_proposed]`` long tensor of ABSOLUTE positions in ``state.x``
        (not block-relative). Sorted by descending proposal confidence so
        that on partial-rejection we can accept the longest matching prefix.
    tokens :
        ``[n_proposed]`` long tensor of proposed token ids, aligned with
        ``positions``.
    confidences :
        ``[n_proposed]`` float tensor of softmax probabilities at each
        proposal. Useful for telemetry / threshold-based rejection.
    """

    positions: "torch.Tensor"
    tokens: "torch.Tensor"
    confidences: "torch.Tensor"

    def __len__(self) -> int:
        return int(self.positions.numel())


def propose(
    block_logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    block_start: int,
    *,
    k: int,
    already_committed_in_step: "torch.Tensor | None" = None,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
) -> Proposal:
    """Pick up to ``k`` masked positions to speculate on, return token proposals.

    Parameters
    ----------
    block_logits :
        ``[B, block_len, V]`` logits for the active block, already shifted
        by the adapter. The engine has these from the regular per-step
        forward — SSD does not add a forward here.
    mask_index :
        ``[B, block_len]`` bool, True at positions still masked at the
        START of this step. (We only propose at positions that are still
        masked AND were NOT just committed by the regular scheduler.)
    block_start :
        Absolute index of the first active-block position in state.x.
        Used to convert block-relative indices to absolute.
    k :
        Maximum number of positions to propose. Will be clamped to
        ``mask_index.sum()`` and to ``len(mask_index) - 1`` (always leave
        at least one masked position so verification can detect drift —
        otherwise an all-True prefix is uninformative).
    already_committed_in_step :
        ``[B, block_len]`` bool of positions just committed by the
        regular scheduler this step. We MUST NOT re-propose those (they
        already have committed tokens written in state.x).
        If None, no exclusion.
    temperature :
        Currently unused (always argmax). Kept in signature so a future
        impl can sample at non-zero temperature; SSD is only strictly
        lossless at temperature == 0.
    confidence_threshold :
        Minimum softmax-max probability required to propose a position.
        0.0 (default) = no filter (paper-style propose). Higher values
        like 0.99 only propose at near-certain positions, where the
        engine's commit order shouldn't matter — the model would predict
        the same token regardless of when in the block it commits. This
        was added in v0.3.0 to bound the order-dependence drift observed
        on Dream-Coder (-25 pp pass@1 with threshold=0.0).

    Returns
    -------
    Proposal
        Up to ``k`` (positions, tokens, confidences) sorted by descending
        confidence, all with confidence ≥ ``confidence_threshold``. May
        be empty if no proposable positions clear the threshold.
    """
    import torch

    del temperature  # see docstring; argmax for now

    if k <= 0:
        return _empty_proposal(block_logits.device)

    B, block_len, V = block_logits.shape
    if B != 1:
        # Multi-sample SSD is a v0.4.0 problem — different samples may want
        # to speculate at different positions, which complicates the
        # batched verify. v0.3.0 ships single-sample only.
        return _empty_proposal(block_logits.device)

    # Filter mask_index to exclude positions just committed.
    proposable = mask_index[0]  # [block_len]
    if already_committed_in_step is not None:
        proposable = proposable & ~already_committed_in_step[0]

    n_proposable = int(proposable.sum())
    if n_proposable == 0:
        return _empty_proposal(block_logits.device)

    # Always leave at least one un-proposed masked position. If everything
    # got proposed, the loop's existing termination check (no masked
    # positions left) would mistake speculation for true completion.
    k_eff = min(k, n_proposable - 1) if n_proposable > 1 else 0
    if k_eff <= 0:
        return _empty_proposal(block_logits.device)

    # Block-relative indices of proposable positions.
    proposable_block_indices = proposable.nonzero(as_tuple=True)[0]  # [n_proposable]

    # Compute per-position confidences (softmax max). Done only on proposable
    # positions to avoid needless work over the full block_len.
    proposable_logits = block_logits[0, proposable_block_indices, :]  # [n_proposable, V]
    probs = torch.softmax(proposable_logits, dim=-1)
    top_p, top_tok = probs.max(dim=-1)  # [n_proposable], [n_proposable]

    # Pick top-k by descending confidence.
    if k_eff >= n_proposable:
        order = torch.argsort(top_p, descending=True)
    else:
        # topk gives indices into top_p, sorted DESC by default.
        _, order = top_p.topk(k_eff, largest=True, sorted=True)

    selected_block_idx = proposable_block_indices[order]
    selected_tokens = top_tok[order]
    selected_confidences = top_p[order]

    # Confidence threshold filter (v0.3.0): drop proposals where the model's
    # top-1 probability is below the threshold. Prevents commit-order drift
    # at uncertain positions — only commit positions where the model would
    # predict the same token regardless of intervening commits.
    if confidence_threshold > 0.0:
        keep = selected_confidences >= confidence_threshold
        selected_block_idx = selected_block_idx[keep]
        selected_tokens = selected_tokens[keep]
        selected_confidences = selected_confidences[keep]
        if selected_block_idx.numel() == 0:
            return _empty_proposal(block_logits.device)

    # Convert block-relative → absolute positions in state.x.
    selected_positions = selected_block_idx + block_start

    return Proposal(
        positions=selected_positions.to(torch.long),
        tokens=selected_tokens.to(torch.long),
        confidences=selected_confidences.float(),
    )


def _empty_proposal(device) -> Proposal:
    import torch

    return Proposal(
        positions=torch.empty(0, dtype=torch.long, device=device),
        tokens=torch.empty(0, dtype=torch.long, device=device),
        confidences=torch.empty(0, dtype=torch.float32, device=device),
    )


def propose_block_level(
    block_logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    block_start: int,
    *,
    confidence_threshold: float,
    max_proposals: int | None = None,
) -> Proposal:
    """Block-level SSD proposal: pick ALL masked positions clearing the
    confidence threshold, sorted by descending confidence.

    Designed for Redesign A (v0.3.0): runs after the block's init forward
    but BEFORE the regular sampler/scheduler. This way SSD gets first pick
    of high-confidence positions, not leftovers from slowfast.

    Differs from ``propose()`` in two ways:
    1. No top-k cap — proposes every position above threshold. The verify
       step's longest-prefix-acceptance handles the rejection naturally.
    2. Leave-one-masked rule still applies (we never propose all positions
       in the block, so the iter loop has something to do at step 1+).

    Parameters
    ----------
    block_logits : ``[B=1, block_len, V]``
        Init forward's logits sliced to active block, already shift_logits'd.
    mask_index : ``[B=1, block_len]`` bool
        True at currently-masked active block positions.
    block_start : int
        Absolute index of the first active-block position in state.x.
    confidence_threshold : float
        Minimum softmax-max for a position to be proposed. 0.95 is a good
        default — order-of-commit doesn't affect the model's prediction
        when it's already 95% sure.
    max_proposals : int | None
        Optional hard cap on proposal count (e.g., to bound verify-forward
        memory at very long blocks). None = no cap, only threshold filters.

    Returns
    -------
    Proposal
        All passing positions, sorted by descending confidence.
    """
    import torch

    if confidence_threshold <= 0.0:
        raise ValueError(
            f"propose_block_level requires confidence_threshold > 0.0; "
            f"got {confidence_threshold}. Use propose() for unbounded "
            f"proposals."
        )

    B, block_len, V = block_logits.shape
    if B != 1:
        return _empty_proposal(block_logits.device)

    proposable = mask_index[0]  # [block_len] bool
    n_proposable = int(proposable.sum())
    if n_proposable <= 1:
        # leave-one-masked rule: need at least 2 masked positions to propose.
        return _empty_proposal(block_logits.device)

    proposable_block_indices = proposable.nonzero(as_tuple=True)[0]  # [n_proposable]
    proposable_logits = block_logits[0, proposable_block_indices, :]  # [n_proposable, V]
    probs = torch.softmax(proposable_logits, dim=-1)
    top_p, top_tok = probs.max(dim=-1)  # [n_proposable]

    # Threshold filter.
    keep = top_p >= confidence_threshold
    if not bool(keep.any()):
        return _empty_proposal(block_logits.device)

    selected_block_idx = proposable_block_indices[keep]
    selected_tokens = top_tok[keep]
    selected_confidences = top_p[keep]

    # Sort by descending confidence so verify's longest-prefix-acceptance
    # gets the most-confident proposals first.
    order = torch.argsort(selected_confidences, descending=True)
    selected_block_idx = selected_block_idx[order]
    selected_tokens = selected_tokens[order]
    selected_confidences = selected_confidences[order]

    # Leave-one-masked: never propose ALL masked positions. Drop the
    # lowest-confidence one if we'd commit everything.
    n_proposed = int(selected_block_idx.numel())
    if n_proposed >= n_proposable:
        # Drop the last (lowest confidence) entry.
        selected_block_idx = selected_block_idx[:-1]
        selected_tokens = selected_tokens[:-1]
        selected_confidences = selected_confidences[:-1]

    # Optional hard cap.
    if max_proposals is not None and selected_block_idx.numel() > max_proposals:
        selected_block_idx = selected_block_idx[:max_proposals]
        selected_tokens = selected_tokens[:max_proposals]
        selected_confidences = selected_confidences[:max_proposals]

    if selected_block_idx.numel() == 0:
        return _empty_proposal(block_logits.device)

    selected_positions = selected_block_idx + block_start
    return Proposal(
        positions=selected_positions.to(torch.long),
        tokens=selected_tokens.to(torch.long),
        confidences=selected_confidences.float(),
    )
