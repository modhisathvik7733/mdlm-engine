"""CPU tests for self-speculative decoding (mdlm_engine.speculative).

No GPU, no real model. We construct stub adapter/state/cache objects
just rich enough for ``propose()`` and ``verify()`` to run, then assert
on the result tensors.

Invariants tested:
    1. ``propose()`` picks high-confidence positions, sorted descending.
    2. ``propose()`` excludes positions just committed by the regular scheduler.
    3. ``propose()`` always leaves at least one masked position un-proposed.
    4. ``propose(k=0)`` is a zero-cost no-op.
    5. ``verify()`` accepts the longest matching argmax prefix.
    6. ``verify()`` rolls back rejected positions in state.x.
    7. ``propose_and_verify()`` orchestrator wires (1)-(6) correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mdlm_engine.speculative import (
    Proposal, propose, propose_and_verify, verify,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubState:
    """Minimal GenerationState for verify() — only `.x` matters."""
    x: torch.Tensor


class _StubCache:
    """Stand-in for DiffusionCache. verify() doesn't read from it; it just
    passes it through to adapter.forward(). We don't even need methods."""
    pass


class _StubAdapter:
    """Adapter whose `.forward` returns predetermined logits.

    Test setup writes the verification logits into ``self.verify_logits``;
    when ``forward()`` is called, that tensor is returned as ``out.logits``
    (shape ``[B, L_full, V]``). The test then asserts on the verify result.
    """

    def __init__(self, verify_logits: torch.Tensor):
        # verify_logits: [B, L_full, V]
        self.verify_logits = verify_logits
        self.forward_call_count = 0

    def forward(self, *, input_ids, attention_mask, position_ids,
                diffusion_cache, use_cache, block_start, block_end, is_init):
        # Stub: ignore all kwargs and return preset logits. Real adapter
        # uses these; the stub just needs the same signature.
        del (input_ids, attention_mask, position_ids, diffusion_cache,
             use_cache, block_start, block_end, is_init)
        self.forward_call_count += 1
        # Pretend to be an AdapterOutput-like object.
        class _Out:
            pass
        out = _Out()
        out.logits = self.verify_logits
        out.past_key_values = None
        return out


# ---------------------------------------------------------------------------
# propose() tests
# ---------------------------------------------------------------------------


def test_propose_picks_highest_confidence_positions():
    """Given block logits where positions 0, 2, 4 are confident at varying
    levels, propose(k=2) returns positions 2 and 4 (top-2 by softmax max).

    Use modest logit gaps so softmax doesn't saturate to 1.0 everywhere —
    we need distinguishable confidences for the descending-sort assertion.
    """
    B, block_len, V = 1, 5, 10
    block_logits = torch.zeros(B, block_len, V)
    # Distinguishable confidences: position 2 most confident, then 4, then 0.
    # softmax(3.0)/sum ≈ 0.74; softmax(2.0) ≈ 0.51; softmax(1.0) ≈ 0.27 at V=10.
    block_logits[0, 0, 7] = 1.0
    block_logits[0, 2, 3] = 3.0
    block_logits[0, 4, 5] = 2.0
    # Positions 1, 3 stay all-zero → uniform, lowest confidence.
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose(block_logits, mask_index, block_start=100, k=2)

    assert len(p) == 2
    # First proposal = position 2 (most confident); second = position 4.
    assert int(p.positions[0]) == 100 + 2
    assert int(p.positions[1]) == 100 + 4
    assert int(p.tokens[0]) == 3
    assert int(p.tokens[1]) == 5
    # Confidences strictly decreasing.
    assert p.confidences[0] > p.confidences[1]


def test_propose_excludes_already_committed():
    """Positions just committed by the regular scheduler must not be
    re-proposed (their tokens are already in state.x)."""
    B, block_len, V = 1, 4, 10
    block_logits = torch.zeros(B, block_len, V)
    # All positions equally confident at different tokens.
    for pos in range(block_len):
        block_logits[0, pos, pos] = 10.0
    mask_index = torch.ones(B, block_len, dtype=torch.bool)
    # Position 1 was just committed — exclude it.
    just_committed = torch.zeros(B, block_len, dtype=torch.bool)
    just_committed[0, 1] = True

    p = propose(block_logits, mask_index, block_start=0, k=4,
                already_committed_in_step=just_committed)

    # Proposable: 0, 2, 3 (3 positions); always-leave-one → k_eff=min(4, 3-1)=2.
    assert len(p) == 2
    assert 1 not in p.positions.tolist()


def test_propose_leaves_at_least_one_masked():
    """If k >= n_masked, k_eff = n_masked - 1. Never propose all of them."""
    B, block_len, V = 1, 3, 10
    block_logits = torch.zeros(B, block_len, V)
    for pos in range(block_len):
        block_logits[0, pos, 5] = 10.0
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose(block_logits, mask_index, block_start=0, k=10)

    # n_masked=3, leave one → 2 proposals.
    assert len(p) == 2


def test_propose_k_zero_is_empty():
    """k=0 returns an empty proposal without computing anything expensive."""
    B, block_len, V = 1, 5, 10
    block_logits = torch.randn(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose(block_logits, mask_index, block_start=0, k=0)

    assert len(p) == 0
    assert p.positions.numel() == 0
    assert p.tokens.numel() == 0


def test_propose_no_masked_positions_returns_empty():
    """If nothing is masked, nothing to propose."""
    B, block_len, V = 1, 5, 10
    block_logits = torch.randn(B, block_len, V)
    mask_index = torch.zeros(B, block_len, dtype=torch.bool)

    p = propose(block_logits, mask_index, block_start=0, k=4)

    assert len(p) == 0


def test_propose_multi_sample_is_no_op():
    """v0.3.0 ships single-sample only; multi-sample returns empty."""
    B, block_len, V = 2, 5, 10  # B=2
    block_logits = torch.randn(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose(block_logits, mask_index, block_start=0, k=4)

    assert len(p) == 0


# ---------------------------------------------------------------------------
# verify() tests
# ---------------------------------------------------------------------------


def _make_proposal(positions, tokens, confidences=None) -> Proposal:
    pos_t = torch.tensor(positions, dtype=torch.long)
    tok_t = torch.tensor(tokens, dtype=torch.long)
    conf_t = (torch.tensor(confidences, dtype=torch.float32)
              if confidences is not None
              else torch.linspace(0.99, 0.5, len(positions)))
    return Proposal(positions=pos_t, tokens=tok_t, confidences=conf_t)


def test_verify_accepts_all_when_argmax_matches():
    """Verifier produces logits whose argmax at each proposed position
    matches the proposed token → all accepted."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _StubCache()

    # Proposal: position 3 → token 7, position 5 → token 11.
    proposal = _make_proposal([3, 5], [7, 11])

    # Build verify_logits so argmax at pos 3 is 7, argmax at pos 5 is 11.
    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 3, 7] = 100.0
    verify_logits[0, 5, 11] = 100.0
    adapter = _StubAdapter(verify_logits)

    result = verify(
        state, cache, adapter, proposal,
        block_start=2, block_end=8,
    )

    assert result.n_accepted == 2
    assert adapter.forward_call_count == 1
    assert result.accepted_positions.tolist() == [3, 5]
    assert result.accepted_tokens.tolist() == [7, 11]
    # state.x reflects the accepted writes.
    assert state.x[0, 3].item() == 7
    assert state.x[0, 5].item() == 11


def test_verify_partial_accept_on_mismatch():
    """First proposal matches (accepted); second mismatches (rejected) →
    accepted prefix length 1, second position rolled back to original."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    state.x[0, 5] = 99  # original sentinel at the to-be-rejected position
    cache = _StubCache()

    proposal = _make_proposal([3, 5], [7, 11])

    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 3, 7] = 100.0   # position 3 matches → accepted
    verify_logits[0, 5, 22] = 100.0  # position 5 argmax=22 ≠ 11 → rejected
    adapter = _StubAdapter(verify_logits)

    result = verify(state, cache, adapter, proposal,
                    block_start=2, block_end=8)

    assert result.n_accepted == 1
    assert result.accepted_positions.tolist() == [3]
    assert result.accepted_tokens.tolist() == [7]
    # state.x: position 3 keeps proposal token 7; position 5 rolled back to 99.
    assert state.x[0, 3].item() == 7
    assert state.x[0, 5].item() == 99


def test_verify_zero_accept_rolls_back_all():
    """All proposals rejected → state.x restored everywhere."""
    L, V = 10, 50
    state = _StubState(x=torch.full((1, L), 42, dtype=torch.long))
    cache = _StubCache()

    proposal = _make_proposal([3, 5], [7, 11])

    verify_logits = torch.zeros(1, L, V)
    # argmax at every position is token 0 → mismatches both proposals.
    adapter = _StubAdapter(verify_logits)

    result = verify(state, cache, adapter, proposal,
                    block_start=2, block_end=8)

    assert result.n_accepted == 0
    assert result.accepted_positions.numel() == 0
    # All positions restored to 42.
    assert (state.x == 42).all()


def test_verify_empty_proposal_short_circuits():
    """Empty proposal → no forward, empty result."""
    L = 10
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _StubCache()
    proposal = Proposal(
        positions=torch.empty(0, dtype=torch.long),
        tokens=torch.empty(0, dtype=torch.long),
        confidences=torch.empty(0, dtype=torch.float32),
    )
    verify_logits = torch.zeros(1, L, 50)
    adapter = _StubAdapter(verify_logits)

    result = verify(state, cache, adapter, proposal,
                    block_start=0, block_end=10)

    assert result.n_accepted == 0
    assert adapter.forward_call_count == 0  # no forward run


def test_verify_first_zero_or_end_helper():
    """Internal helper: first False index, or len if all True."""
    from mdlm_engine.speculative.verify import _first_zero_or_end

    assert _first_zero_or_end(torch.tensor([True, True, True])) == 3
    assert _first_zero_or_end(torch.tensor([True, False, True])) == 1
    assert _first_zero_or_end(torch.tensor([False, True, True])) == 0
    assert _first_zero_or_end(torch.tensor([True, True, False])) == 2
    assert _first_zero_or_end(torch.tensor([], dtype=torch.bool)) == 0


# ---------------------------------------------------------------------------
# propose_and_verify() orchestrator
# ---------------------------------------------------------------------------


def test_propose_and_verify_full_acceptance():
    """High-confidence proposal whose argmax matches verifier → full accept."""
    L, V, block_len = 12, 50, 4
    block_start = 5
    block_end = 9

    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _StubCache()

    block_logits = torch.zeros(1, block_len, V)
    # Highly confident at block-rel position 0 → token 7 (= absolute pos 5)
    # and block-rel 2 → token 11 (= absolute pos 7).
    block_logits[0, 0, 7] = 50.0
    block_logits[0, 2, 11] = 50.0
    mask_index = torch.ones(1, block_len, dtype=torch.bool)

    # Verifier returns same argmax — full accept.
    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 5, 7] = 100.0
    verify_logits[0, 7, 11] = 100.0
    adapter = _StubAdapter(verify_logits)

    result = propose_and_verify(
        state, cache, adapter, block_logits, mask_index,
        block_start=block_start, block_end=block_end,
        k=2,
    )

    # 4 masked, leave-one rule → k_eff = min(2, 4-1) = 2.
    assert result.n_accepted == 2
    assert sorted(result.accepted_positions.tolist()) == [5, 7]


def test_propose_and_verify_k_zero_is_no_op():
    """k=0 → no proposal, no forward, empty result."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _StubCache()

    block_logits = torch.randn(1, 4, V)
    mask_index = torch.ones(1, 4, dtype=torch.bool)
    adapter = _StubAdapter(torch.zeros(1, L, V))

    result = propose_and_verify(
        state, cache, adapter, block_logits, mask_index,
        block_start=0, block_end=4,
        k=0,
    )

    assert result.n_accepted == 0
    assert adapter.forward_call_count == 0


def test_propose_and_verify_no_masked_is_no_op():
    """No masked positions → empty proposal → no forward."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _StubCache()

    block_logits = torch.randn(1, 4, V)
    mask_index = torch.zeros(1, 4, dtype=torch.bool)
    adapter = _StubAdapter(torch.zeros(1, L, V))

    result = propose_and_verify(
        state, cache, adapter, block_logits, mask_index,
        block_start=0, block_end=4,
        k=2,
    )

    assert result.n_accepted == 0
    assert adapter.forward_call_count == 0
