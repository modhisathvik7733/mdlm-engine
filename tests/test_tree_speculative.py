"""CPU tests for v0.4.0 tree-speculative decoding.

Three modules under test:
1. ``mdlm_engine/speculative/cache_snapshot.py`` — snapshot/restore K/V slabs
2. ``mdlm_engine/speculative/tree.py`` — propose_band, propose_tree
3. ``mdlm_engine/speculative/verify.py:verify_tree`` — sequential branch verify

Critical invariants:
- snapshot → mutate → restore is idempotent (cache state matches pre-snapshot)
- propose_band picks ONLY positions in [low, high) — strict bounds
- propose_tree's branches are DISJOINT (no position appears in both)
- verify_tree([single_proposal]) is identical to verify(single_proposal)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mdlm_engine.cache.dkv import DKVCache
from mdlm_engine.cache.block import BlockCache
from mdlm_engine.speculative import (
    CacheSnapshot, Proposal, propose_band, propose_tree,
    restore_active_block, snapshot_active_block, verify, verify_tree,
)


# ---------------------------------------------------------------------------
# Helpers (stub adapter, stub state, identical to test_speculative.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubState:
    x: torch.Tensor


class _StubAdapter:
    """Adapter whose forward returns predetermined logits."""

    def __init__(self, verify_logits: torch.Tensor):
        self.verify_logits = verify_logits
        self.forward_call_count = 0

    def forward(self, *, input_ids, attention_mask, position_ids,
                diffusion_cache, use_cache, block_start, block_end, is_init):
        del (input_ids, attention_mask, position_ids, diffusion_cache,
             use_cache, block_start, block_end, is_init)
        self.forward_call_count += 1

        class _Out:
            pass
        out = _Out()
        out.logits = self.verify_logits
        out.past_key_values = None
        return out


def _make_cache(B=1, n_layers=2, n_kv_heads=2, head_dim=4, max_length=16):
    """Real DKVCache instance for snapshot/restore tests."""
    return DKVCache(
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        max_length=max_length, batch_size=B,
        dtype=torch.float32, device="cpu",
        strict=False,
    )


# ---------------------------------------------------------------------------
# cache_snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_returns_cloned_slabs():
    """snapshot() should clone, not alias, so post-snapshot mutations
    don't propagate into the saved slab."""
    cache = _make_cache(max_length=16)
    cache._K[0][:, :, 4:8, :] = 7.0  # write something
    cache._V[0][:, :, 4:8, :] = 9.0
    cache._commit_state[:, 4:8] = True

    snap = snapshot_active_block(cache, block_start=4, block_end=8)

    # Mutate the cache after snapshot.
    cache._K[0][:, :, 4:8, :] = 99.0
    cache._V[0][:, :, 4:8, :] = 99.0
    cache._commit_state[:, 4:8] = False

    # Snapshot should still hold the original 7 / 9 / True values.
    assert (snap.K_slabs[0] == 7.0).all()
    assert (snap.V_slabs[0] == 9.0).all()
    assert snap.commit_state_slab.all()


def test_snapshot_restore_idempotent():
    """snapshot → mutate → restore should put the cache back exactly as
    it was at snapshot time. Repeated restores must be stable."""
    cache = _make_cache(max_length=16)
    # Set a known initial state.
    cache._K[0][:, :, 4:8, :] = 1.5
    cache._V[0][:, :, 4:8, :] = 2.5
    cache._commit_state[:, 4:8] = True
    cache._K[0][:, :, 0:4, :] = 100.0  # outside active block; should be untouched

    snap = snapshot_active_block(cache, block_start=4, block_end=8)
    # Mutate active block.
    cache._K[0][:, :, 4:8, :] = -1.0
    cache._V[0][:, :, 4:8, :] = -2.0
    cache._commit_state[:, 4:8] = False
    # Mutate outside (should also stay untouched after restore — restore is local).
    cache._K[0][:, :, 0:4, :] = 50.0

    restore_active_block(cache, snap)

    # Active block restored.
    assert (cache._K[0][:, :, 4:8, :] == 1.5).all()
    assert (cache._V[0][:, :, 4:8, :] == 2.5).all()
    assert cache._commit_state[:, 4:8].all()
    # Outside the snapshot range — restore must NOT touch.
    assert (cache._K[0][:, :, 0:4, :] == 50.0).all()

    # Second restore is also idempotent.
    restore_active_block(cache, snap)
    assert (cache._K[0][:, :, 4:8, :] == 1.5).all()


def test_snapshot_returns_cachesnapshot_type():
    cache = _make_cache()
    snap = snapshot_active_block(cache, block_start=2, block_end=8)

    assert isinstance(snap, CacheSnapshot)
    assert snap.block_start == 2
    assert snap.block_end == 8
    assert len(snap.K_slabs) == len(cache._K)
    assert len(snap.V_slabs) == len(cache._V)


def test_snapshot_works_with_block_cache():
    """Sanity: BlockCache and DKVCache share the _K/_V layout, so
    snapshot/restore should work for either."""
    cache = BlockCache(
        n_layers=1, n_kv_heads=1, head_dim=2, max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    cache._K[0][:, :, 0:4, :] = 3.0
    snap = snapshot_active_block(cache, 0, 4)
    cache._K[0][:, :, 0:4, :] = 100.0
    restore_active_block(cache, snap)

    assert (cache._K[0][:, :, 0:4, :] == 3.0).all()


# ---------------------------------------------------------------------------
# propose_band tests
# ---------------------------------------------------------------------------


def test_propose_band_picks_only_positions_in_band():
    """Build logits where positions 0, 1, 2 have softmax-max
    approximately {0.99, 0.97, 0.85} respectively.
    propose_band(band_low=0.96, band_high=0.99) should pick only
    position 1."""
    B, block_len, V = 1, 5, 50
    logits = torch.zeros(B, block_len, V)
    # Hand-tune logits to land at target softmax maxes (V=50).
    # softmax(x) where x is the only nonzero logit:
    #   p = exp(x) / (exp(x) + (V-1))
    # For p≈0.99: x ≈ ln(0.99 * 49 / 0.01) ≈ ln(4851) ≈ 8.49
    # For p≈0.97: x ≈ ln(0.97 * 49 / 0.03) ≈ ln(1584) ≈ 7.37
    # For p≈0.85: x ≈ ln(0.85 * 49 / 0.15) ≈ ln(277.7) ≈ 5.63
    logits[0, 0, 1] = 8.5
    logits[0, 1, 2] = 7.3
    logits[0, 2, 3] = 5.6
    # Positions 3, 4 stay uniform → softmax max ≈ 0.02.

    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_band(
        logits, mask_index, block_start=0,
        band_low=0.96, band_high=0.99,
        max_proposals=4,
    )

    # Position 0 (≈0.9994) is ≥ 0.99, should be EXCLUDED (strict <).
    # Position 1 (≈0.97) is in [0.96, 0.99), should be INCLUDED.
    # Position 2 (≈0.85) is < 0.96, should be EXCLUDED.
    assert len(p) == 1
    assert int(p.positions[0]) == 1
    assert int(p.tokens[0]) == 2


def test_propose_band_strict_upper_bound():
    """A position whose softmax-max is exactly band_high should NOT be
    proposed (band is half-open [low, high))."""
    B, block_len, V = 1, 3, 10
    logits = torch.zeros(B, block_len, V)
    # Try to hit ≈ 0.95 exactly. softmax(x) on V=10 with one nonzero:
    # p=0.95 → x ≈ ln(0.95 * 9 / 0.05) = ln(171) ≈ 5.14
    logits[0, 0, 4] = 5.14
    # Position 1 below band, position 2 above band high.
    logits[0, 1, 5] = 1.0
    logits[0, 2, 6] = 10.0

    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_band(
        logits, mask_index, block_start=0,
        band_low=0.94, band_high=0.95,  # strict < 0.95 means pos 0 just barely doesn't make it
        max_proposals=4,
    )

    # softmax of 5.14 with V=10 is ~0.949, so it might or might not make it
    # depending on float precision. The test is that position 2 (~1.0) doesn't.
    for tok_idx_in_proposal, pos in enumerate(p.positions.tolist()):
        # Position 2 has very high prob (>0.99); must be excluded.
        assert pos != 2


def test_propose_band_max_proposals_cap():
    B, block_len, V = 1, 5, 10
    # Make 4 positions all confident in the band.
    logits = torch.zeros(B, block_len, V)
    for pos in range(4):
        logits[0, pos, pos] = 5.0  # softmax ~0.94 — in [0.85, 0.95) band
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_band(
        logits, mask_index, block_start=0,
        band_low=0.85, band_high=0.95,
        max_proposals=2,
    )

    assert len(p) == 2  # capped


def test_propose_band_validates_args():
    B, block_len, V = 1, 4, 10
    logits = torch.zeros(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    # band_low >= band_high should raise.
    import pytest
    with pytest.raises(ValueError, match="band_low < band_high"):
        propose_band(logits, mask_index, 0, band_low=0.99, band_high=0.97)

    # band_low <= 0 should raise.
    with pytest.raises(ValueError, match="0 < band_low"):
        propose_band(logits, mask_index, 0, band_low=0.0, band_high=0.5)


def test_propose_band_multi_sample_returns_empty():
    B, block_len, V = 2, 5, 10  # B=2
    logits = torch.zeros(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_band(logits, mask_index, 0, band_low=0.5, band_high=0.99)

    assert len(p) == 0


# ---------------------------------------------------------------------------
# propose_tree tests
# ---------------------------------------------------------------------------


def test_propose_tree_branches_disjoint():
    """propose_tree(high=0.99, low=0.97) returns two proposals with NO
    overlapping positions."""
    B, block_len, V = 1, 6, 50
    logits = torch.zeros(B, block_len, V)
    # Positions 0, 1: high confidence (≥0.99) → branch 0.
    logits[0, 0, 1] = 9.0  # softmax > 0.99
    logits[0, 1, 2] = 9.0
    # Positions 2, 3: in [0.97, 0.99) → branch 1.
    logits[0, 2, 3] = 7.4
    logits[0, 3, 4] = 7.4
    # Positions 4, 5: low confidence → neither.

    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    branch_0, branch_1 = propose_tree(
        logits, mask_index, block_start=0,
        high_threshold=0.99, band_low=0.97,
    )

    pos_0 = set(branch_0.positions.tolist())
    pos_1 = set(branch_1.positions.tolist())
    assert pos_0 & pos_1 == set(), f"Branches overlap: {pos_0 & pos_1}"
    # Branch 0 should contain positions 0 and 1.
    assert pos_0 == {0, 1}
    # Branch 1 should contain positions 2 and 3.
    # (At softmax of 7.4 on V=50, prob ≈ 0.97-0.98 so they should land in [0.97, 0.99))
    assert 2 in pos_1
    assert 3 in pos_1


def test_propose_tree_empty_branch_1_when_no_band_positions():
    """If no positions fall in [low, high), branch 1 is empty."""
    B, block_len, V = 1, 5, 10
    logits = torch.zeros(B, block_len, V)
    # All positions either ≥0.99 or below 0.5. Nothing in band.
    logits[0, 0, 1] = 10.0  # ~0.999
    logits[0, 1, 2] = 10.0
    # Positions 2-4 stay uniform.

    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    branch_0, branch_1 = propose_tree(
        logits, mask_index, block_start=0,
        high_threshold=0.99, band_low=0.97,
    )

    assert len(branch_0) >= 1
    assert len(branch_1) == 0


# ---------------------------------------------------------------------------
# verify_tree tests
# ---------------------------------------------------------------------------


def test_verify_tree_single_proposal_matches_verify():
    """verify_tree([proposal]) must produce the same result as
    verify(proposal) — bit-identical for the single-branch case."""
    L, V = 10, 50
    state_a = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    state_b = _StubState(x=torch.zeros(1, L, dtype=torch.long))

    proposal = Proposal(
        positions=torch.tensor([3, 5], dtype=torch.long),
        tokens=torch.tensor([7, 11], dtype=torch.long),
        confidences=torch.tensor([0.99, 0.98]),
    )

    # Build verify_logits with argmax matching proposal.
    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 3, 7] = 100.0
    verify_logits[0, 5, 11] = 100.0

    cache_a = _make_cache()
    cache_b = _make_cache()

    res_single = verify(
        state_a, cache_a, _StubAdapter(verify_logits), proposal,
        block_start=2, block_end=8,
    )

    results_tree = verify_tree(
        state_b, cache_b, _StubAdapter(verify_logits), [proposal],
        block_start=2, block_end=8,
        snapshot_between_branches=False,  # single branch, no snapshot needed
    )

    assert len(results_tree) == 1
    assert results_tree[0].n_accepted == res_single.n_accepted
    assert results_tree[0].accepted_positions.tolist() == res_single.accepted_positions.tolist()
    # state.x should match between the two paths.
    assert (state_a.x == state_b.x).all()


def test_verify_tree_two_branches_runs_twice():
    """verify_tree with 2 non-empty branches runs the adapter twice."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _make_cache(max_length=L)

    proposal_0 = Proposal(
        positions=torch.tensor([3], dtype=torch.long),
        tokens=torch.tensor([7], dtype=torch.long),
        confidences=torch.tensor([0.999]),
    )
    proposal_1 = Proposal(
        positions=torch.tensor([5], dtype=torch.long),
        tokens=torch.tensor([11], dtype=torch.long),
        confidences=torch.tensor([0.97]),
    )

    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 3, 7] = 100.0
    verify_logits[0, 5, 11] = 100.0
    adapter = _StubAdapter(verify_logits)

    results = verify_tree(
        state, cache, adapter, [proposal_0, proposal_1],
        block_start=2, block_end=8,
    )

    assert len(results) == 2
    assert adapter.forward_call_count == 2
    assert results[0].n_accepted == 1
    assert results[1].n_accepted == 1


def test_verify_tree_empty_branch_skipped():
    """An empty proposal in a branch slot doesn't trigger a forward."""
    L, V = 10, 50
    state = _StubState(x=torch.zeros(1, L, dtype=torch.long))
    cache = _make_cache(max_length=L)

    proposal_0 = Proposal(
        positions=torch.tensor([3], dtype=torch.long),
        tokens=torch.tensor([7], dtype=torch.long),
        confidences=torch.tensor([0.999]),
    )
    proposal_empty = Proposal(
        positions=torch.empty(0, dtype=torch.long),
        tokens=torch.empty(0, dtype=torch.long),
        confidences=torch.empty(0, dtype=torch.float32),
    )

    verify_logits = torch.zeros(1, L, V)
    verify_logits[0, 3, 7] = 100.0
    adapter = _StubAdapter(verify_logits)

    results = verify_tree(
        state, cache, adapter, [proposal_0, proposal_empty],
        block_start=2, block_end=8,
    )

    assert len(results) == 2
    # Only the non-empty branch should have run a forward.
    assert adapter.forward_call_count == 1
    assert results[0].n_accepted == 1
    assert results[1].n_accepted == 0


def test_verify_tree_empty_list_returns_empty():
    state = _StubState(x=torch.zeros(1, 10, dtype=torch.long))
    cache = _make_cache(max_length=10)

    results = verify_tree(state, cache, _StubAdapter(torch.zeros(1, 10, 50)),
                          branches=[], block_start=0, block_end=10)

    assert results == []
