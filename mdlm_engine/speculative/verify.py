"""Self-speculative verification step — one extra forward, accept prefix.

Given a Proposal from ``propose()``, write the proposed tokens into a
TEMPORARY copy of ``state.x``, run one adapter forward, and check whether
the model's argmax at each proposed position agrees with what we wrote.

Key invariants:
- At ``temperature == 0``, the regular engine would commit token T at
  position p iff the model's argmax at p (under the prevailing context)
  is T. SSD verification reproduces that exact predicate. Net effect:
  identical token sequence, fewer forwards.
- We accept the longest *prefix* of proposals (in confidence order) that
  matches; rejection of proposal i invalidates all subsequent proposals
  because their context assumed i was committed. This matches arxiv
  2510.04147 §3.2 ("Verification").
- The verification forward MUTATES the cache (PATH A's
  ``replace_position`` rewrites K/V at active block positions). To
  preserve the cache state for the engine's next step, we either (a)
  snapshot K/V before verification and restore on partial reject, or
  (b) accept the cache mutation and let the next step recompute. Option
  (b) is simpler and correct because the next step would have run a
  forward with the same active block anyway — the cache K/V SSD wrote
  is exactly what step+1 would have written.

That second invariant means SSD with PATH A "for free" merges its
verification forward into the normal step pipeline: a verified-and-
accepted SSD round saves one full step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from mdlm_engine.adapters.base import ModelAdapter
    from mdlm_engine.cache.base import DiffusionCache
    from mdlm_engine.core.state import GenerationState
    from mdlm_engine.speculative.propose import Proposal


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one SSD verification round.

    Attributes
    ----------
    n_accepted :
        Length of the accepted prefix (in proposal order, descending
        confidence). 0 if all proposals rejected; len(proposal) if all
        accepted.
    accepted_positions :
        Subset of ``proposal.positions`` that the verifier confirmed.
        Same dtype as proposal.positions.
    accepted_tokens :
        Aligned with ``accepted_positions``.
    """

    n_accepted: int
    accepted_positions: "torch.Tensor"
    accepted_tokens: "torch.Tensor"


def verify(
    state: "GenerationState",
    cache: "DiffusionCache",
    adapter: "ModelAdapter",
    proposal: "Proposal",
    *,
    block_start: int,
    block_end: int,
    attn_mask_full: "torch.Tensor | str | None" = None,
    position_ids_full: "torch.Tensor | None" = None,
) -> VerificationResult:
    """Run one verification forward; return accepted prefix.

    Mutates state.x to reflect accepted tokens. Does NOT mutate cache
    state — the caller (loop.py) decides how to update the cache after
    accepting (typically: call cache.commit() for accepted positions).

    Parameters
    ----------
    state :
        Engine generation state. ``state.x`` is patched in-place at the
        proposed positions for the verification forward, then either
        kept (on accept) or rolled back (on reject).
    cache :
        DiffusionCache. The verification forward will write through to
        the cache via PATH A's in-place K/V replace; this is correct
        because subsequent commits on accepted positions would have
        produced the same K/V anyway.
    adapter :
        ModelAdapter; we call ``adapter.forward(..., is_init=False)``.
    proposal :
        Output of ``propose()`` — positions/tokens/confidences sorted
        by descending confidence.
    block_start, block_end :
        Active block bounds (absolute, inclusive start, exclusive end).
    attn_mask_full :
        Full-sequence attention_mask the engine uses for adapter.forward.
    position_ids_full :
        Full-sequence position_ids the engine uses for adapter.forward.

    Returns
    -------
    VerificationResult
    """
    import torch

    if len(proposal) == 0:
        return _empty_result(state.x.device)

    # Write proposed tokens into a temp copy of state.x. Restore on reject.
    original_at_positions = state.x[0, proposal.positions].clone()
    state.x[0, proposal.positions] = proposal.tokens

    try:
        # Verification forward — same code path as the engine's iter step
        # (PATH A iter or PATH C full forward, depending on adapter).
        out = adapter.forward(
            input_ids=state.x,
            attention_mask=attn_mask_full,
            position_ids=position_ids_full,
            diffusion_cache=cache,
            use_cache=True,
            block_start=block_start,
            block_end=block_end,
            is_init=False,
        )
    except Exception:
        # Restore on any failure — caller treats as zero-acceptance.
        state.x[0, proposal.positions] = original_at_positions
        raise

    # Adapter's logits are full-shape (zeros outside active block on PATH A
    # iter). Extract argmax at proposal positions and compare with what we
    # wrote.
    full_logits = out.logits  # [B, L_full, V] — adapter already shift_logits'd
    proposed_argmax = full_logits[0, proposal.positions, :].argmax(dim=-1)  # [n_proposed]
    matches = (proposed_argmax == proposal.tokens)  # [n_proposed]

    # Longest-prefix acceptance: find first mismatch index.
    # `matches` is sorted by descending proposal confidence.
    n_accepted = int(_first_zero_or_end(matches))

    if n_accepted == 0:
        # No proposals accepted — restore original state.x at all proposed
        # positions (verification's cache mutation stays; that's still
        # consistent because state.x is now identical to its pre-SSD value
        # at those positions, and the next regular step will overwrite
        # K/V there with the right tokens).
        state.x[0, proposal.positions] = original_at_positions
        return _empty_result(state.x.device)

    # Partial or full acceptance: keep tokens at accepted positions, restore
    # at rejected ones.
    if n_accepted < len(proposal):
        rejected_positions = proposal.positions[n_accepted:]
        # Slice of original_at_positions aligned with rejected.
        state.x[0, rejected_positions] = original_at_positions[n_accepted:]

    accepted_positions = proposal.positions[:n_accepted]
    accepted_tokens = proposal.tokens[:n_accepted]
    return VerificationResult(
        n_accepted=n_accepted,
        accepted_positions=accepted_positions,
        accepted_tokens=accepted_tokens,
    )


def _first_zero_or_end(matches: "torch.Tensor") -> int:
    """Return the index of the first False in ``matches``, or len(matches)
    if all True. Equivalent to the longest True-prefix length."""
    import torch

    if matches.numel() == 0:
        return 0
    if bool(matches.all()):
        return int(matches.numel())
    # First False index = argmax over (1 - matches.int()) — argmax returns
    # the FIRST occurrence of the max value when there are ties, which is
    # exactly what we want.
    return int((~matches).int().argmax().item())


def _empty_result(device) -> VerificationResult:
    import torch

    return VerificationResult(
        n_accepted=0,
        accepted_positions=torch.empty(0, dtype=torch.long, device=device),
        accepted_tokens=torch.empty(0, dtype=torch.long, device=device),
    )


def verify_tree(
    state: "GenerationState",
    cache: "DiffusionCache",
    adapter: "ModelAdapter",
    branches: "list[Proposal]",
    *,
    block_start: int,
    block_end: int,
    attn_mask_full: "torch.Tensor | str | None" = None,
    position_ids_full: "torch.Tensor | None" = None,
    snapshot_between_branches: bool = True,
) -> "list[VerificationResult]":
    """Run verify per branch sequentially. Optionally snapshot/restore the
    cache's active-block K/V between branches so each branch sees a
    consistent pre-state.

    v0.4.0 tree-spec semantics:
    - Branch 0 verifies first; its accepted commits MUTATE state.x and
      cache K/V at the accepted positions.
    - Branch 1+ verify AFTER branch 0's commits are in state.x. They see
      richer committed context, which may push some borderline positions
      above their argmax-match threshold.
    - Cache K/V at the active block is restored between branches IF
      ``snapshot_between_branches=True`` (default), so branch 1's verify
      starts from the same K/V state as a fresh single-branch run would.
      Branch 0's accepted-position writes still propagate forward via
      state.x — only K/V is restored, not the committed tokens.

    Why restore K/V but not state.x: PATH A's iter forward computes K/V
    at active block from state.x's current values. With branch 0's
    committed tokens in state.x, the next forward will recompute K/V
    correctly anyway. The restore prevents branch 0's full *speculative*
    forward (including REJECTED proposed tokens) from contaminating
    branch 1's logits.

    Returns
    -------
    list[VerificationResult]
        One result per branch, in the same order as ``branches``.
        Empty branches (``len(proposal) == 0``) get a no-op result with
        ``n_accepted=0``.
    """
    from mdlm_engine.speculative.cache_snapshot import (
        snapshot_active_block, restore_active_block,
    )

    if not branches:
        return []

    results: list[VerificationResult] = []

    # Take snapshot once before any branch runs. Restored before each
    # branch >= 1 so they all see the same pre-verify cache state.
    if snapshot_between_branches and len(branches) > 1:
        pre_verify_snapshot = snapshot_active_block(cache, block_start, block_end)
    else:
        pre_verify_snapshot = None

    for branch_idx, proposal in enumerate(branches):
        if branch_idx > 0 and pre_verify_snapshot is not None:
            # Restore cache K/V to pre-branch-0 state. state.x retains
            # branch-0 commits — those are the "borrowed context" that
            # gives branch 1 its potential speedup.
            restore_active_block(cache, pre_verify_snapshot)

        if len(proposal) == 0:
            results.append(_empty_result(state.x.device))
            continue

        result = verify(
            state=state,
            cache=cache,
            adapter=adapter,
            proposal=proposal,
            block_start=block_start,
            block_end=block_end,
            attn_mask_full=attn_mask_full,
            position_ids_full=position_ids_full,
        )
        results.append(result)

    return results
