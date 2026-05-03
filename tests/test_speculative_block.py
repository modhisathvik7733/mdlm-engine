"""CPU tests for ``propose_block_level`` (Redesign A entry point).

Differences from ``propose()`` (covered in test_speculative.py):
- No top-k cap. Picks ALL positions clearing the confidence threshold.
- Designed for step-0 use on init forward's logits, before the sampler.
- Same leave-one-masked rule (so the regular iter loop has work left).

Invariants tested:
1. Picks all positions ≥ threshold, sorted descending.
2. Leave-one-masked rule still applies.
3. Empty when no positions clear threshold.
4. Empty when only 0-1 positions are masked.
5. Multi-sample (B>1) returns empty (single-sample only in v0.3.0).
6. Confidence sort verified.
7. ``max_proposals`` cap honored.
8. Threshold=0.0 raises (use ``propose()`` instead).
"""
from __future__ import annotations

import pytest
import torch

from mdlm_engine.speculative import Proposal, propose_block_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confident_logits(B, block_len, V, position_token_pairs, magnitude=10.0):
    """Build [B, block_len, V] logits where each (pos, token) gets a logit
    ``magnitude``; everything else is 0. Higher magnitude → softmax max
    closer to 1.0.
    """
    logits = torch.zeros(B, block_len, V)
    for pos, tok in position_token_pairs:
        logits[0, pos, tok] = magnitude
    return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_propose_block_level_picks_all_above_threshold():
    """4 confident positions clear 0.9; 1 below; result: 3 (leave-one rule)
    for sufficiently confident logits. With strong magnitude, all 4 clear
    → 3 proposed (leave one)."""
    B, block_len, V = 1, 5, 10
    # All 4 first positions have a token with logit 10 → softmax ≈ 0.9999
    logits = _confident_logits(
        B, block_len, V,
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        magnitude=10.0,
    )
    # Position 4 stays uniform → softmax max ≈ 0.1
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=100,
        confidence_threshold=0.9,
    )

    # 4 positions clear threshold but leave-one-masked rule drops 1.
    # Wait — n_proposable=5, n_proposed=4, so 4 < 5; rule doesn't trigger.
    # BUT re-check: after threshold filter, we have 4 proposals out of 5
    # masked. n_proposed=4 < n_proposable=5 → no drop.
    # However leave-one rule says NEVER propose all masked. We have 5
    # masked, propose 4 — that's fine (one left unproposed = pos 4).
    assert len(p) == 4
    # Sorted DESC by confidence (all 4 should have similar high confidence;
    # order among them is arbitrary but assert sorted).
    assert torch.all(p.confidences[:-1] >= p.confidences[1:])


def test_propose_block_level_leaves_one_when_all_clear():
    """All 5 positions clear threshold. Must leave 1 unproposed."""
    B, block_len, V = 1, 5, 10
    logits = _confident_logits(
        B, block_len, V,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        magnitude=10.0,
    )
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
    )

    # 5 masked, all clear threshold, leave-one rule → 4 proposed.
    assert len(p) == 4


def test_propose_block_level_empty_below_threshold():
    """No positions clear threshold → empty proposal."""
    B, block_len, V = 1, 5, 10
    # All uniform → softmax max ≈ 0.1, well below 0.9
    logits = torch.zeros(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
    )

    assert len(p) == 0


def test_propose_block_level_empty_when_one_masked():
    """Leave-one-masked rule: only 1 position masked → empty proposal
    (would need to leave at least 1 — can't propose any)."""
    B, block_len, V = 1, 5, 10
    logits = _confident_logits(B, block_len, V, [(2, 7)], magnitude=10.0)
    mask_index = torch.zeros(B, block_len, dtype=torch.bool)
    mask_index[0, 2] = True

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
    )

    assert len(p) == 0


def test_propose_block_level_excludes_unmasked_positions():
    """Confident position that's NOT masked must not be proposed."""
    B, block_len, V = 1, 5, 10
    # Make positions 1 and 3 confident.
    logits = _confident_logits(B, block_len, V,
                               [(1, 7), (3, 9)], magnitude=10.0)
    # But mark only positions 0, 1 as masked. Position 3 is unmasked.
    mask_index = torch.zeros(B, block_len, dtype=torch.bool)
    mask_index[0, 0] = True
    mask_index[0, 1] = True

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
    )

    # Only position 1 clears threshold AND is masked. Position 3 is
    # confident but unmasked. Position 0 is masked but uniform (low conf).
    # 2 masked total, 1 clears threshold, 0 left-one drop → 1 proposed.
    assert len(p) == 1
    assert int(p.positions[0]) == 1
    assert int(p.tokens[0]) == 7


def test_propose_block_level_sort_descending():
    """Proposals must come back sorted by descending confidence."""
    B, block_len, V = 1, 4, 10
    logits = torch.zeros(B, block_len, V)
    # Three confident positions with DIFFERENT magnitudes.
    # softmax(3.0) on V=10 ≈ 0.74; (5.0) ≈ 0.95; (8.0) ≈ 0.99.
    logits[0, 0, 1] = 3.0
    logits[0, 1, 2] = 5.0
    logits[0, 2, 3] = 8.0
    # Position 3 stays uniform.
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=10,
        confidence_threshold=0.6,
    )

    # softmax over V=10 with single non-zero logit:
    #   logit 3.0 → ~0.69 (clears 0.6); 5.0 → ~0.94; 8.0 → ~0.99.
    # All 3 clear 0.6. 4 masked, 3 proposed → leave-one-rule doesn't drop.
    assert len(p) == 3
    # Sorted desc: pos 2 (0.99), pos 1 (0.95), pos 0 (0.74).
    assert int(p.positions[0]) == 10 + 2
    assert int(p.positions[1]) == 10 + 1
    assert int(p.positions[2]) == 10 + 0


def test_propose_block_level_max_proposals_cap():
    """``max_proposals`` clamps the result count after threshold filtering."""
    B, block_len, V = 1, 6, 10
    # All 5 positions confident at varied magnitudes; pos 5 stays uniform.
    logits = _confident_logits(B, block_len, V,
                               [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                               magnitude=10.0)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
        max_proposals=2,
    )

    # 6 masked, 5 clear threshold, leave-one drops to 4, cap drops to 2.
    assert len(p) == 2


def test_propose_block_level_multi_sample_returns_empty():
    """v0.3.0 ships single-sample only; B>1 returns empty."""
    B, block_len, V = 2, 5, 10
    logits = _confident_logits(B, block_len, V,
                               [(0, 1), (1, 2)], magnitude=10.0)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=0,
        confidence_threshold=0.9,
    )

    assert len(p) == 0


def test_propose_block_level_zero_threshold_raises():
    """threshold=0 means no filter — but propose_block_level is designed
    around threshold-as-primary-filter. Use propose() for unbounded."""
    B, block_len, V = 1, 5, 10
    logits = torch.randn(B, block_len, V)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    with pytest.raises(ValueError, match="confidence_threshold"):
        propose_block_level(
            logits, mask_index, block_start=0,
            confidence_threshold=0.0,
        )


def test_propose_block_level_returns_proposal_type():
    """Output is a Proposal namedtuple/dataclass with positions, tokens,
    confidences tensors of equal length."""
    B, block_len, V = 1, 5, 10
    logits = _confident_logits(B, block_len, V,
                               [(0, 7), (2, 3)], magnitude=10.0)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)

    p = propose_block_level(
        logits, mask_index, block_start=50,
        confidence_threshold=0.9,
    )

    assert isinstance(p, Proposal)
    assert p.positions.shape == p.tokens.shape == p.confidences.shape
    assert p.positions.dtype == torch.long
    assert p.tokens.dtype == torch.long
    assert p.confidences.dtype == torch.float32
