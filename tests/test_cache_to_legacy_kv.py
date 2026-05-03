"""Phase 2 Day 1 tests for ``DiffusionCache.to_legacy_kv()``.

The method materializes the cache as HF's ``past_key_values`` shape — a list
of ``(K, V)`` per layer. This is what every Dream/Llama/Qwen-family model's
forward expects when you pass ``past_key_values=...``.

CPU-only.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# BlockCache
# ---------------------------------------------------------------------------


def test_block_cache_to_legacy_kv_returns_list_of_per_layer_tuples():
    """BlockCache.to_legacy_kv() should return n_layers tuples of (K, V)."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=3, n_kv_heads=4, head_dim=8,
        max_length=16, batch_size=2,
        dtype=torch.float32, device="cpu",
    )
    pkv = cache.to_legacy_kv()

    assert isinstance(pkv, list)
    assert len(pkv) == 3                # n_layers
    for layer in pkv:
        assert isinstance(layer, tuple)
        assert len(layer) == 2           # (K, V)
        K, V = layer
        # Shape contract: [B, n_kv_heads, max_length, head_dim].
        assert K.shape == (2, 4, 16, 8)
        assert V.shape == (2, 4, 16, 8)


def test_block_cache_to_legacy_kv_shares_storage_with_internal_tensors():
    """The returned tensors are views/aliases of the cache's internal _K/_V —
    NOT copies. Mutations to the cache reflect in the legacy view, and
    vice versa (the model is going to write into these tensors directly)."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )

    # Write something into the cache; verify the legacy view reflects it.
    positions = torch.tensor([3], dtype=torch.long)
    K_new = torch.full((1, 2, 1, 4), 7.0)
    V_new = torch.full((1, 2, 1, 4), 9.0)
    cache.replace_at(0, positions, K_new, V_new)

    pkv = cache.to_legacy_kv()
    K_view, V_view = pkv[0]
    assert K_view[0, 0, 3, 0].item() == 7.0
    assert V_view[0, 0, 3, 0].item() == 9.0

    # Now the other direction: mutate the legacy view, verify cache sees it.
    # (This is exactly what the model does when it returns past_key_values.)
    K_view[0, 0, 5, 0] = 42.0
    K_internal, _ = cache.read_full(0)
    assert K_internal[0, 0, 5, 0].item() == 42.0


def test_block_cache_to_legacy_kv_layers_are_independent():
    """Mutating layer 0's K/V shouldn't affect layer 1."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=2, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )

    positions = torch.tensor([0], dtype=torch.long)
    cache.replace_at(0, positions, torch.full((1, 2, 1, 4), 5.0), torch.full((1, 2, 1, 4), 5.0))

    pkv = cache.to_legacy_kv()
    K0, _ = pkv[0]
    K1, _ = pkv[1]
    assert K0[0, 0, 0, 0].item() == 5.0
    assert K1[0, 0, 0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# DKVCache
# ---------------------------------------------------------------------------


def test_dkv_cache_to_legacy_kv_inherits_default_impl():
    """DKVCache uses the base-class default impl; same shape contract."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=2, n_kv_heads=4, head_dim=8,
        max_length=16, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    pkv = cache.to_legacy_kv()

    assert len(pkv) == 2
    for K, V in pkv:
        assert K.shape == (1, 4, 16, 8)
        assert V.shape == (1, 4, 16, 8)


def test_dkv_cache_to_legacy_kv_reflects_writes():
    """Same alias-of-internal-storage contract as BlockCache."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    positions = torch.tensor([2], dtype=torch.long)
    cache.replace_at(0, positions, torch.full((1, 2, 1, 4), 3.0), torch.full((1, 2, 1, 4), 3.0))

    pkv = cache.to_legacy_kv()
    K_view, V_view = pkv[0]
    assert K_view[0, 0, 2, 0].item() == 3.0
    assert V_view[0, 0, 2, 0].item() == 3.0


# ---------------------------------------------------------------------------
# NoOpCache — special case
# ---------------------------------------------------------------------------


def test_noop_cache_to_legacy_kv_returns_none_per_layer():
    """NoOpCache returns ``[None] * n_layers`` — signals "no cache, recompute".

    HF's modeling code checks for ``past_key_values is None`` (or a layer entry
    being None) and skips the cache concat path. So returning None per layer
    is the explicit "no cache" signal the model expects.
    """
    from mdlm_engine.cache.base import NoOpCache

    cache = NoOpCache(
        n_layers=3, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    pkv = cache.to_legacy_kv()
    assert isinstance(pkv, list)
    assert len(pkv) == 3
    assert all(layer is None for layer in pkv)


# ---------------------------------------------------------------------------
# Default-impl fallthrough catches subclasses that forget to override
# ---------------------------------------------------------------------------


def test_to_legacy_kv_raises_for_subclass_without_K_V_storage():
    """A custom DiffusionCache that doesn't expose _K/_V should get a clear
    error from the default to_legacy_kv. Catches "subclass forgot to override"
    bugs at the call site instead of producing a confusing AttributeError."""
    from mdlm_engine.cache.base import DiffusionCache

    class _SparseLikeCache(DiffusionCache):
        # Implements the abstract methods but stores K/V in a custom shape.
        def replace_at(self, layer, positions, K, V):
            pass
        def read_full(self, layer):
            return torch.empty(0), torch.empty(0)
        def commit(self, positions):
            pass
        def commit_state(self):
            return torch.zeros(self.batch_size, self.max_length, dtype=torch.bool)
        def reset(self):
            pass

    cache = _SparseLikeCache(
        n_layers=1, n_kv_heads=1, head_dim=1,
        max_length=1, batch_size=1, dtype=torch.float32, device="cpu",
    )
    with pytest.raises(NotImplementedError, match="_SparseLikeCache.*_K/_V"):
        cache.to_legacy_kv()


# ---------------------------------------------------------------------------
# Smoke: list length matches the n_layers attribute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 4, 32])
def test_to_legacy_kv_length_matches_n_layers(n_layers):
    """Whatever n_layers we allocate, that's how many entries we get."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=n_layers, n_kv_heads=1, head_dim=2,
        max_length=4, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    assert len(cache.to_legacy_kv()) == n_layers
