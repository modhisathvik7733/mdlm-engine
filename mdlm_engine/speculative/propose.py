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
) -> Proposal:
    """Pick ``k`` masked positions to speculate on, return token proposals.

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

    Returns
    -------
    Proposal
        Up to ``k`` (positions, tokens, confidences) sorted by descending
        confidence. May be empty if there are no proposable positions.
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
