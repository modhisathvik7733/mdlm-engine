"""Unit tests for `mdlm_engine.cache.block.BlockCache`.

CPU-only. Verifies the `DiffusionCache` ABC contract end-to-end:
    - replace_at writes K/V at the correct positions (1D and 2D positions)
    - read_full returns the workspace tensors
    - commit/commit_state record frozen positions
    - reset is cheap (zeroes commit state, doesn't reallocate)
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def small_cache():
    """A tiny BlockCache for fast unit tests."""
    from mdlm_engine.cache.block import BlockCache

    return BlockCache(
        n_layers=2,
        n_kv_heads=4,
        head_dim=8,
        max_length=16,
        batch_size=2,
        dtype=torch.float32,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Construction + reset
# ---------------------------------------------------------------------------


def test_block_cache_initial_state(small_cache):
    """Newly constructed cache: zeros K/V, all-False commit state."""
    K, V = small_cache.read_full(0)
    assert K.shape == (2, 4, 16, 8)
    assert V.shape == (2, 4, 16, 8)
    assert (K == 0).all()
    assert (V == 0).all()

    assert small_cache.commit_state().shape == (2, 16)
    assert small_cache.commit_state().dtype == torch.bool
    assert not small_cache.commit_state().any()


def test_reset_clears_commit_state_but_not_tensors(small_cache):
    """reset() is for cheap reuse — commit state goes to all-False, K/V stays
    (it'll be overwritten by the next replace_at anyway)."""
    positions = torch.tensor([0, 5, 10], dtype=torch.long)
    K = torch.randn(2, 4, 3, 8)
    V = torch.randn(2, 4, 3, 8)
    small_cache.replace_at(0, positions, K, V)
    small_cache.commit(positions)

    assert small_cache.commit_state().any()

    small_cache.reset()
    assert not small_cache.commit_state().any()
    # K/V is intentionally not cleared. The engine doesn't read uncommitted
    # positions, so leftover values are harmless. Confirm we can still
    # read them (no crash) — the actual values don't matter.
    K_after, V_after = small_cache.read_full(0)
    assert K_after.shape == (2, 4, 16, 8)


# ---------------------------------------------------------------------------
# replace_at — 1D positions (broadcast across batch)
# ---------------------------------------------------------------------------


def test_replace_at_1d_positions(small_cache):
    positions = torch.tensor([0, 5, 10], dtype=torch.long)
    K = torch.randn(2, 4, 3, 8)
    V = torch.randn(2, 4, 3, 8)

    small_cache.replace_at(0, positions, K, V)
    K_full, V_full = small_cache.read_full(0)

    # Positions we wrote should match exactly.
    assert torch.allclose(K_full[:, :, 0, :], K[:, :, 0, :])
    assert torch.allclose(K_full[:, :, 5, :], K[:, :, 1, :])
    assert torch.allclose(K_full[:, :, 10, :], K[:, :, 2, :])
    assert torch.allclose(V_full[:, :, 5, :], V[:, :, 1, :])

    # Positions we didn't write should remain zero.
    assert (K_full[:, :, 1, :] == 0).all()
    assert (K_full[:, :, 15, :] == 0).all()


def test_replace_at_1d_overwrites_previous_writes(small_cache):
    """A second replace_at at the same position should overwrite the first."""
    positions = torch.tensor([3], dtype=torch.long)
    K1 = torch.ones(2, 4, 1, 8)
    V1 = torch.ones(2, 4, 1, 8)
    small_cache.replace_at(0, positions, K1, V1)

    K2 = torch.full((2, 4, 1, 8), 7.0)
    V2 = torch.full((2, 4, 1, 8), 9.0)
    small_cache.replace_at(0, positions, K2, V2)

    K_full, V_full = small_cache.read_full(0)
    assert (K_full[:, :, 3, :] == 7.0).all()
    assert (V_full[:, :, 3, :] == 9.0).all()


# ---------------------------------------------------------------------------
# replace_at — 2D positions (per-sample)
# ---------------------------------------------------------------------------


def test_replace_at_2d_positions(small_cache):
    """Per-sample positions: sample 0 writes to [0,1,2], sample 1 to [10,11,12]."""
    positions = torch.tensor(
        [
            [0, 1, 2],
            [10, 11, 12],
        ],
        dtype=torch.long,
    )
    K = torch.randn(2, 4, 3, 8)
    V = torch.randn(2, 4, 3, 8)
    small_cache.replace_at(0, positions, K, V)

    K_full, V_full = small_cache.read_full(0)
    # Sample 0 should have K written at positions 0, 1, 2.
    assert torch.allclose(K_full[0, :, 0, :], K[0, :, 0, :])
    assert torch.allclose(K_full[0, :, 1, :], K[0, :, 1, :])
    assert torch.allclose(K_full[0, :, 2, :], K[0, :, 2, :])
    # Sample 0 should have ZERO at positions 10, 11, 12 (only sample 1 wrote there).
    assert (K_full[0, :, 10, :] == 0).all()
    # Sample 1 should have K written at positions 10, 11, 12.
    assert torch.allclose(K_full[1, :, 10, :], K[1, :, 0, :])
    assert torch.allclose(K_full[1, :, 12, :], K[1, :, 2, :])
    # Sample 1 should have ZERO at positions 0, 1, 2.
    assert (K_full[1, :, 0, :] == 0).all()


# ---------------------------------------------------------------------------
# Layer isolation
# ---------------------------------------------------------------------------


def test_layers_are_independent(small_cache):
    """Writing to layer 0 must not affect layer 1."""
    positions = torch.tensor([0], dtype=torch.long)
    K = torch.full((2, 4, 1, 8), 5.0)
    V = torch.full((2, 4, 1, 8), 5.0)
    small_cache.replace_at(0, positions, K, V)

    K_layer0, _ = small_cache.read_full(0)
    K_layer1, _ = small_cache.read_full(1)
    assert (K_layer0[:, :, 0, :] == 5.0).all()
    assert (K_layer1[:, :, 0, :] == 0.0).all()


# ---------------------------------------------------------------------------
# commit / commit_state
# ---------------------------------------------------------------------------


def test_commit_marks_positions(small_cache):
    positions = torch.tensor([2, 4, 6], dtype=torch.long)
    small_cache.commit(positions)

    state = small_cache.commit_state()
    # 1D positions broadcast across batch.
    assert state[:, 2].all()
    assert state[:, 4].all()
    assert state[:, 6].all()
    assert not state[:, 0].any()
    assert not state[:, 8].any()


def test_commit_2d_positions_per_sample(small_cache):
    """Per-sample commits: sample 0 commits [1,2], sample 1 commits [10,11]."""
    positions = torch.tensor([[1, 2], [10, 11]], dtype=torch.long)
    small_cache.commit(positions)

    state = small_cache.commit_state()
    assert state[0, 1] and state[0, 2]
    assert not state[0, 10] and not state[0, 11]  # sample 0 didn't commit there
    assert state[1, 10] and state[1, 11]
    assert not state[1, 1] and not state[1, 2]


def test_num_committed_property(small_cache):
    """`num_committed` returns per-sample counts of frozen positions."""
    positions = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]], dtype=torch.long)
    small_cache.commit(positions)
    n = small_cache.num_committed
    assert n.shape == (2,)
    assert n.tolist() == [4, 4]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_replace_at_rejects_shape_mismatch(small_cache):
    """positions and K must agree on n_pos."""
    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    K = torch.randn(2, 4, 5, 8)  # 5 positions in K but only 3 in positions
    V = torch.randn(2, 4, 5, 8)
    with pytest.raises(ValueError, match="K.shape"):
        small_cache.replace_at(0, positions, K, V)


def test_replace_at_rejects_3d_positions(small_cache):
    """positions must be 1D or 2D — 3D raises."""
    positions = torch.zeros(2, 3, 1, dtype=torch.long)
    K = torch.zeros(2, 4, 3, 8)
    V = torch.zeros(2, 4, 3, 8)
    with pytest.raises(ValueError, match="1D or 2D"):
        small_cache.replace_at(0, positions, K, V)


def test_commit_rejects_3d_positions(small_cache):
    positions = torch.zeros(2, 3, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="1D"):
        small_cache.commit(positions)


# ---------------------------------------------------------------------------
# Module re-export
# ---------------------------------------------------------------------------


def test_block_cache_is_exported_from_cache_package():
    """Public API: `from mdlm_engine.cache import BlockCache` must work."""
    from mdlm_engine.cache import BlockCache as Imported
    from mdlm_engine.cache.block import BlockCache as Direct

    assert Imported is Direct
