"""Phase 2 Day 2 tests for ``DiffusionCache.update_from_model_output``.

The method bridges from a model's returned ``past_key_values`` back into
the cache. Two paths exercised:

1. **Alias-based in-place** (the primary Phase 2 path): ``to_legacy_kv()``
   returns aliases of the cache's internal ``_K``/``_V``. The model writes
   through those aliases in-place (fast_dllm's ``dual_cache=True`` pattern).
   The cache is updated automatically — no explicit call needed. Tests
   verify the alias contract.

2. **Explicit update** (the fallback): for models that return non-aliased
   ``past_key_values``, ``update_from_model_output(layer, (K, V), positions)``
   writes the model's K/V into the cache. Tests verify both BlockCache (uses
   default impl that goes through replace_at) and DKVCache (overrides to
   bypass the strict committed-slot check, since the model is authoritative).

CPU-only.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Path 1: alias-based in-place updates (the primary Phase 2 flow)
# ---------------------------------------------------------------------------


def test_alias_based_inplace_write_through_to_legacy_kv():
    """Simulate the dual_cache=True flow: get past_key_values, mutate in-place,
    cache should reflect the write without any explicit update call."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=2, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    pkv = cache.to_legacy_kv()

    # The "model" writes new K/V at position 3 of layer 0 (this is what
    # fast_dllm's `past_key[:, replace_indices] = key_states` does, with
    # past_key being our alias of cache._K[0]).
    K_layer0, V_layer0 = pkv[0]
    K_layer0[:, :, 3, :] = 7.0
    V_layer0[:, :, 3, :] = 9.0

    # Cache should see those values via read_full (no update call needed).
    K_internal, V_internal = cache.read_full(0)
    assert (K_internal[:, :, 3, :] == 7.0).all()
    assert (V_internal[:, :, 3, :] == 9.0).all()

    # Layer 1 should be untouched.
    K1, V1 = cache.read_full(1)
    assert (K1 == 0.0).all()
    assert (V1 == 0.0).all()


def test_alias_based_works_with_dkv_cache():
    """Same alias-based contract for DKVCache."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    pkv = cache.to_legacy_kv()
    K, V = pkv[0]
    K[:, :, 5, :] = 3.0
    V[:, :, 5, :] = 4.0

    K_internal, V_internal = cache.read_full(0)
    assert (K_internal[:, :, 5, :] == 3.0).all()
    assert (V_internal[:, :, 5, :] == 4.0).all()


# ---------------------------------------------------------------------------
# Path 2a: explicit update via BlockCache default impl (goes through replace_at)
# ---------------------------------------------------------------------------


def test_block_cache_update_from_model_output_writes_at_positions():
    """Default impl: pass new K/V at positions; cache reflects them."""
    from mdlm_engine.cache.block import BlockCache

    cache = BlockCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    K_new = torch.full((1, 2, 2, 4), 5.0)
    V_new = torch.full((1, 2, 2, 4), 6.0)
    positions = torch.tensor([1, 4], dtype=torch.long)

    cache.update_from_model_output(layer=0, past_key_values_layer=(K_new, V_new), positions=positions)

    K_full, V_full = cache.read_full(0)
    assert (K_full[:, :, 1, :] == 5.0).all()
    assert (K_full[:, :, 4, :] == 5.0).all()
    assert (V_full[:, :, 1, :] == 6.0).all()
    # Positions we didn't write should still be zero.
    assert (K_full[:, :, 0, :] == 0.0).all()
    assert (K_full[:, :, 7, :] == 0.0).all()


# ---------------------------------------------------------------------------
# Path 2b: DKVCache overrides update_from_model_output to bypass strict check
# ---------------------------------------------------------------------------


def test_dkv_cache_update_from_model_output_bypasses_strict_check():
    """Strict mode: replace_at on a committed slot raises (catches engine bugs).
    But update_from_model_output is the MODEL writing back its authoritative
    values — should bypass the strict check.

    Concretely: commit position 3, then have the "model" return new K/V at
    position 3. update_from_model_output should accept that without raising,
    even though replace_at(layer, [3], ...) would raise."""
    from mdlm_engine.cache.dkv import CommittedSlotWriteError, DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
        strict=True,
    )
    # First write + commit at position 3.
    init_K = torch.full((1, 2, 1, 4), 1.0)
    init_V = torch.full((1, 2, 1, 4), 1.0)
    pos = torch.tensor([3], dtype=torch.long)
    cache.replace_at(0, pos, init_K, init_V)
    cache.commit(pos)

    # Sanity: replace_at WOULD raise on this committed slot.
    with pytest.raises(CommittedSlotWriteError):
        cache.replace_at(0, pos, init_K * 2, init_V * 2)

    # But update_from_model_output (the "model is authoritative" path) accepts.
    new_K = torch.full((1, 2, 1, 4), 99.0)
    new_V = torch.full((1, 2, 1, 4), 88.0)
    cache.update_from_model_output(layer=0, past_key_values_layer=(new_K, new_V), positions=pos)

    K_full, V_full = cache.read_full(0)
    assert (K_full[:, :, 3, :] == 99.0).all()
    assert (V_full[:, :, 3, :] == 88.0).all()


def test_dkv_cache_update_from_model_output_2d_positions():
    """Per-sample positions in update_from_model_output (multi-sample batch)."""
    from mdlm_engine.cache.dkv import DKVCache

    cache = DKVCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=2,
        dtype=torch.float32, device="cpu",
        strict=False,    # don't care about commit overlap here
    )
    # Sample 0 writes at positions [0, 1]; sample 1 writes at [4, 5].
    K_new = torch.randn(2, 2, 2, 4)
    V_new = torch.randn(2, 2, 2, 4)
    positions = torch.tensor([[0, 1], [4, 5]], dtype=torch.long)

    cache.update_from_model_output(
        layer=0, past_key_values_layer=(K_new, V_new), positions=positions,
    )

    K_full, V_full = cache.read_full(0)
    # Sample 0's writes appear at its positions, not sample 1's.
    assert torch.allclose(K_full[0, :, 0, :], K_new[0, :, 0, :])
    assert torch.allclose(K_full[0, :, 1, :], K_new[0, :, 1, :])
    assert (K_full[0, :, 4, :] == 0.0).all()  # sample 0 didn't write here
    # Sample 1's writes appear at its positions.
    assert torch.allclose(K_full[1, :, 4, :], K_new[1, :, 0, :])
    assert torch.allclose(K_full[1, :, 5, :], K_new[1, :, 1, :])
    assert (K_full[1, :, 0, :] == 0.0).all()


# ---------------------------------------------------------------------------
# Edge: NoOpCache update is a no-op
# ---------------------------------------------------------------------------


def test_noop_cache_update_from_model_output_is_safe_noop():
    """NoOpCache stores nothing; updates are silently dropped (no crash)."""
    from mdlm_engine.cache.base import NoOpCache

    cache = NoOpCache(
        n_layers=1, n_kv_heads=2, head_dim=4,
        max_length=8, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    K = torch.zeros(1, 2, 1, 4)
    V = torch.zeros(1, 2, 1, 4)
    pos = torch.tensor([0], dtype=torch.long)
    # Should not raise.
    cache.update_from_model_output(layer=0, past_key_values_layer=(K, V), positions=pos)
