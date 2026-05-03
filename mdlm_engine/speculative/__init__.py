"""Self-speculative decoding for masked diffusion LMs (arxiv 2510.04147).

Public API:
    propose_and_verify(state, cache, adapter, block_logits, mask_index,
                       block_start, block_end, k, *, attn_mask_full=...,
                       position_ids_full=...) -> VerificationResult

Algorithm summary:
    1. propose(): pick k masked positions with highest argmax confidence
       from the current step's already-computed logits. No extra forward.
    2. verify(): write proposed tokens into state.x, run ONE additional
       adapter forward, accept the longest matching argmax prefix.
    3. Caller (core/loop.py) commits accepted positions in the cache and
       continues the next step.

At temperature == 0 this is exactly equivalent to running the engine for
N more regular steps where N = n_accepted, but uses 1 forward instead
of N. Speedup ratio = (1 + n_accepted) / 2 (one verify forward replaces
n_accepted regular forwards on accept). Paper claim: 3.46x lossless.

For temperature > 0 the equivalence is approximate (the engine's
sampler would have rolled different RNG); we still accept the argmax
prefix as a "good enough" proposal — the model trained for diffusion
already prefers these tokens at these positions.

NOT shipped in v0.2.x. v0.3.0 wires this in via
``mdlm_engine.core.loop.LoopConfig.speculative_k``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mdlm_engine.speculative.propose import (
    Proposal, propose, propose_block_level,
)
from mdlm_engine.speculative.verify import VerificationResult, verify

if TYPE_CHECKING:
    import torch

    from mdlm_engine.adapters.base import ModelAdapter
    from mdlm_engine.cache.base import DiffusionCache
    from mdlm_engine.core.state import GenerationState


def propose_and_verify(
    state: "GenerationState",
    cache: "DiffusionCache",
    adapter: "ModelAdapter",
    block_logits: "torch.Tensor",
    mask_index: "torch.Tensor",
    *,
    block_start: int,
    block_end: int,
    k: int,
    already_committed_in_step: "torch.Tensor | None" = None,
    attn_mask_full: "torch.Tensor | str | None" = None,
    position_ids_full: "torch.Tensor | None" = None,
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
) -> VerificationResult:
    """Run one round of speculation. Returns the accepted prefix.

    If k == 0 or no positions can be proposed, returns an empty result
    without running any extra forward (zero-cost no-op).

    Caller is responsible for:
    - Calling ``cache.commit(result.accepted_positions)`` after this
      returns, if it wants those positions to count as committed for
      future steps.
    - Updating its own pass counters / progress prints with
      ``result.n_accepted``.
    """
    proposal = propose(
        block_logits=block_logits,
        mask_index=mask_index,
        block_start=block_start,
        k=k,
        already_committed_in_step=already_committed_in_step,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )
    if len(proposal) == 0:
        from mdlm_engine.speculative.verify import _empty_result
        return _empty_result(state.x.device)

    return verify(
        state=state,
        cache=cache,
        adapter=adapter,
        proposal=proposal,
        block_start=block_start,
        block_end=block_end,
        attn_mask_full=attn_mask_full,
        position_ids_full=position_ids_full,
    )


__all__ = [
    "Proposal",
    "VerificationResult",
    "propose",
    "propose_block_level",
    "verify",
    "propose_and_verify",
]
