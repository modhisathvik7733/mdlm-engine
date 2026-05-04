"""Snapshot/restore active-block K/V slabs for tree-spec verification.

v0.4.0 tree spec runs K=2 verification forwards per step. Branch 1's
verify pass needs to start from the same cache state as a single-branch
SSD run would see post-branch-0 — but our cache K/V is shared and gets
mutated in-place by every PATH A iter forward. So we snapshot the
active-block K/V slab before branch 0's verify and restore between
branches.

Memory cost: for Dream-7B at block_length=32, snapshot is
``28 layers × 2 (K,V) × [1, 4, 32, 128] × bf16 ≈ 1.8 MB`` per step.
Negligible vs the 15 GB VRAM headroom and vs the ~12 ms per forward.

The snapshot intentionally does NOT cover positions outside the active
block — those K/V are unchanged by PATH A's iter pass (the model only
recomputes K/V at masked positions, which by construction live in the
active block). Restoring the active-block slab is sufficient to roll
back any verify-induced mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from mdlm_engine.cache.base import DiffusionCache


@dataclass(frozen=True)
class CacheSnapshot:
    """Slice of cache state covering the active block only.

    Attributes
    ----------
    K_slabs :
        ``list[Tensor]`` of length ``n_layers``. Each tensor has shape
        ``[B, n_kv_heads, block_length, head_dim]`` — the active-block
        slice of the original ``cache._K[layer]``. Cloned at snapshot
        time so subsequent in-place writes to the cache don't alias.
    V_slabs :
        Same shape as ``K_slabs``, cloned from ``cache._V``.
    commit_state_slab :
        ``[B, block_length]`` bool slice of ``cache._commit_state`` for
        the active block. Cloned.
    block_start :
        Inclusive left edge of the active block this snapshot covers.
    block_end :
        Exclusive right edge.
    """

    K_slabs: "list[torch.Tensor]"
    V_slabs: "list[torch.Tensor]"
    commit_state_slab: "torch.Tensor"
    block_start: int
    block_end: int


def snapshot_active_block(
    cache: "DiffusionCache",
    block_start: int,
    block_end: int,
) -> CacheSnapshot:
    """Clone cache K/V slabs and commit-state row for the active block.

    Constraint: the cache must expose ``_K``, ``_V`` (lists of 4D tensors
    indexed ``[B, n_kv_heads, L_max, head_dim]``) and ``_commit_state``
    (``[B, L_max]`` bool). All concrete caches in mdlm_engine satisfy
    this (BlockCache, DKVCache; NoOpCache is excluded — caller's
    responsibility to gate on cache type).

    Raises ``AttributeError`` if the cache subclass doesn't store K/V
    in the expected layout — same as ``DiffusionCache.to_legacy_kv()``
    fallback at ``cache/base.py``.
    """
    if not hasattr(cache, "_K") or not hasattr(cache, "_V"):
        raise AttributeError(
            f"{type(cache).__name__} doesn't expose _K/_V; "
            f"snapshot_active_block requires the standard 4D cache layout."
        )

    K_slabs = [
        cache._K[layer][:, :, block_start:block_end, :].clone()
        for layer in range(len(cache._K))
    ]
    V_slabs = [
        cache._V[layer][:, :, block_start:block_end, :].clone()
        for layer in range(len(cache._V))
    ]
    commit_state_slab = cache._commit_state[:, block_start:block_end].clone()

    return CacheSnapshot(
        K_slabs=K_slabs,
        V_slabs=V_slabs,
        commit_state_slab=commit_state_slab,
        block_start=block_start,
        block_end=block_end,
    )


def restore_active_block(
    cache: "DiffusionCache",
    snapshot: CacheSnapshot,
) -> None:
    """Write snapshot K/V slabs and commit-state row back into the cache.

    Mutates ``cache._K[layer]``, ``cache._V[layer]``, and
    ``cache._commit_state`` in place via ``copy_`` (no allocation).

    Idempotent: ``restore(snapshot(cache))`` followed by another
    ``restore(snapshot(cache))`` produces the same cache state. Tested
    in ``tests/test_tree_speculative.py::test_snapshot_restore_idempotent``.
    """
    block_start = snapshot.block_start
    block_end = snapshot.block_end

    for layer in range(len(cache._K)):
        cache._K[layer][:, :, block_start:block_end, :].copy_(snapshot.K_slabs[layer])
        cache._V[layer][:, :, block_start:block_end, :].copy_(snapshot.V_slabs[layer])
    cache._commit_state[:, block_start:block_end].copy_(snapshot.commit_state_slab)


__all__ = ["CacheSnapshot", "snapshot_active_block", "restore_active_block"]
