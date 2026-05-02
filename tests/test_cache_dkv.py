"""Unit tests for `mdlm_engine.cache.dkv.DKVCache` (CPU-only)."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def dkv():
    from mdlm_engine.cache.dkv import DKVCache

    return DKVCache(
        n_layers=2, n_kv_heads=4, head_dim=8,
        max_length=16, batch_size=2,
        dtype=torch.float32, device="cpu",
    )


def test_basic_write_and_read(dkv):
    positions = torch.tensor([0, 5, 10], dtype=torch.long)
    K = torch.randn(2, 4, 3, 8)
    V = torch.randn(2, 4, 3, 8)
    dkv.replace_at(0, positions, K, V)
    K_full, V_full = dkv.read_full(0)
    assert torch.allclose(K_full[:, :, 0, :], K[:, :, 0, :])
    assert torch.allclose(K_full[:, :, 5, :], K[:, :, 1, :])
    assert torch.allclose(V_full[:, :, 10, :], V[:, :, 2, :])


def test_strict_mode_blocks_committed_overwrite(dkv):
    """Default strict=True: overwriting a committed slot raises."""
    from mdlm_engine.cache.dkv import CommittedSlotWriteError

    positions = torch.tensor([3], dtype=torch.long)
    K = torch.ones(2, 4, 1, 8)
    V = torch.ones(2, 4, 1, 8)
    dkv.replace_at(0, positions, K, V)
    dkv.commit(positions)

    with pytest.raises(CommittedSlotWriteError, match="committed slots"):
        dkv.replace_at(0, positions, K * 2, V * 2)


def test_strict_false_only_warns():
    """strict=False downgrades the violation to a RuntimeWarning."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
        strict=False,
    )
    positions = torch.tensor([0], dtype=torch.long)
    K = torch.ones(1, 2, 1, 4)
    V = torch.ones(1, 2, 1, 4)
    cache.replace_at(0, positions, K, V)
    cache.commit(positions)

    with pytest.warns(RuntimeWarning, match="committed slots"):
        cache.replace_at(0, positions, K * 2, V * 2)


def test_no_overlap_check_when_writing_uncommitted(dkv):
    """Writing to UN-committed positions is always fine, even if other
    positions are committed."""
    pos_committed = torch.tensor([0, 1], dtype=torch.long)
    pos_fresh = torch.tensor([5, 6], dtype=torch.long)
    K = torch.randn(2, 4, 2, 8)
    V = torch.randn(2, 4, 2, 8)
    dkv.replace_at(0, pos_committed, K, V)
    dkv.commit(pos_committed)
    # Writing to 5, 6 must not raise — they are not committed.
    dkv.replace_at(0, pos_fresh, K, V)


def test_one_step_delay_defers_commit():
    """one_step_delay=True: commits don't apply until the NEXT commit() call."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
        one_step_delay=True,
    )
    pos_a = torch.tensor([0], dtype=torch.long)
    pos_b = torch.tensor([1], dtype=torch.long)

    cache.commit(pos_a)
    # pos_a is pending, not yet in commit_state
    assert not cache.commit_state()[0, 0]

    cache.commit(pos_b)
    # Now pos_a has flushed to commit_state, pos_b is pending
    assert cache.commit_state()[0, 0]
    assert not cache.commit_state()[0, 1]

    cache.commit(torch.tensor([], dtype=torch.long))  # empty tick to flush
    assert cache.commit_state()[0, 1]


def test_reset_clears_both_commit_and_pending(dkv):
    """reset() must clear commit_state AND any pending commits."""
    pos = torch.tensor([0, 1, 2], dtype=torch.long)
    dkv.commit(pos)
    assert dkv.commit_state().any()
    dkv.reset()
    assert not dkv.commit_state().any()
    # _pending_commits is internal, but accessible for the test.
    assert not dkv._pending_commits.any()


def test_2d_positions_per_sample(dkv):
    """[B, n_pos] positions: each sample writes to its own positions."""
    positions = torch.tensor([[0, 1], [10, 11]], dtype=torch.long)
    K = torch.randn(2, 4, 2, 8)
    V = torch.randn(2, 4, 2, 8)
    dkv.replace_at(0, positions, K, V)
    K_full, _ = dkv.read_full(0)
    assert torch.allclose(K_full[0, :, 0, :], K[0, :, 0, :])
    assert torch.allclose(K_full[1, :, 11, :], K[1, :, 1, :])
    # Verify NO cross-write: sample 0's slot 11 should still be zero.
    assert (K_full[0, :, 11, :] == 0).all()


def test_committed_overlap_detection_2d(dkv):
    """Strict mode also catches per-sample committed overlaps."""
    from mdlm_engine.cache.dkv import CommittedSlotWriteError

    pos = torch.tensor([[0], [5]], dtype=torch.long)
    K = torch.randn(2, 4, 1, 8)
    V = torch.randn(2, 4, 1, 8)
    dkv.replace_at(0, pos, K, V)
    dkv.commit(pos)

    # Try to overwrite sample 0's slot 0 — should raise even though
    # sample 1's slot 0 is uncommitted.
    overlap_pos = torch.tensor([[0], [0]], dtype=torch.long)
    with pytest.raises(CommittedSlotWriteError):
        dkv.replace_at(0, overlap_pos, K, V)


def test_dkv_cache_exported_from_package():
    from mdlm_engine.cache import CommittedSlotWriteError, DKVCache
    from mdlm_engine.cache.dkv import CommittedSlotWriteError as Direct1
    from mdlm_engine.cache.dkv import DKVCache as Direct2

    assert DKVCache is Direct2
    assert CommittedSlotWriteError is Direct1


def test_dkv_inherits_diffusion_cache():
    from mdlm_engine.cache.base import DiffusionCache
    from mdlm_engine.cache.dkv import DKVCache

    assert issubclass(DKVCache, DiffusionCache)
