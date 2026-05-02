"""BlockCache — Phase-1 baseline KV-cache for masked diffusion LMs.

Allocates a workspace K/V tensor of shape ``[B, n_kv_heads, max_length, head_dim]``
per layer, plus a ``[B, max_length]`` bool commit-state mask.

Semantics
---------
- ``replace_at(layer, positions, K, V)`` writes K/V into the workspace slots.
- ``read_full(layer)`` returns the full ``(K, V)`` tensors for the model to attend
  against. The model's attention uses the *whole* L_max range; uncommitted
  positions whose K/V was overwritten by the most recent ``replace_at`` are
  fresh; committed positions keep whatever was last written.
- ``commit(positions)`` records that those positions are frozen — the engine
  uses this to decide which positions to recompute on the next step (i.e.
  skip committed ones) but the cache does NOT enforce immutability. A scheduler
  bug that re-writes a committed slot is caught by the equivalence test, not
  by the cache.

This is the equivalent of fast_dllm's *non*-``dual_cache`` path, generalized
to be model-agnostic. It's the Phase-1 baseline; ``DKVCache`` (Day 3) is the
faster, position-indexed variant that drives the Phase-1 speedup gate.

Design notes
------------
- Memory: at 7B / GQA8 / 4096-ctx / bf16, one layer's K *or* V is
  ``B × 8 × 4096 × 128 × 2 = 8 MB``; for 32 layers that's 32 × 2 × 8 MB ≈ **512 MB**
  of cache at batch 1. Comfortable on a 24 GB+ GPU.
- The K/V tensors are pre-allocated once in ``__init__`` and reused across
  generations via ``reset()``. No allocations in the hot path.
- All methods are batch-aware. ``positions`` may be ``[n_pos]`` (broadcast) or
  ``[B, n_pos]`` (per-sample), in line with the ABC.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mdlm_engine.cache.base import DiffusionCache

if TYPE_CHECKING:
    pass


class BlockCache(DiffusionCache):
    """Block-prefix KV cache (Phase-1 baseline)."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_length=max_length,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        # Pre-allocate K/V workspace per layer. Single layout: [B, n_kv, L, d_h].
        self._K: list[torch.Tensor] = [
            torch.zeros(
                batch_size, n_kv_heads, max_length, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(n_layers)
        ]
        self._V: list[torch.Tensor] = [
            torch.zeros(
                batch_size, n_kv_heads, max_length, head_dim,
                dtype=dtype, device=device,
            )
            for _ in range(n_layers)
        ]
        # commit_state: True = position has frozen K/V the engine should not recompute.
        self._commit_state = torch.zeros(
            batch_size, max_length, dtype=torch.bool, device=device,
        )

    # ------------------------------------------------------------------
    # Storage operations (mandatory ABC contract)
    # ------------------------------------------------------------------

    def replace_at(
        self,
        layer: int,
        positions: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Write K/V into the cache slots indexed by `positions`.

        Parameters
        ----------
        layer : int
            Index in ``[0, n_layers)``.
        positions :
            Either ``[n_pos]`` long (same positions for every sample in the batch)
            or ``[B, n_pos]`` long (per-sample positions). Values must be in
            ``[0, max_length)``.
        K, V :
            Each ``[B, n_kv_heads, n_pos, head_dim]``.
        """
        self._validate_positions(positions, K)
        if positions.ndim == 1:
            # Broadcast: same positions for every sample.
            self._K[layer].index_copy_(2, positions, K)
            self._V[layer].index_copy_(2, positions, V)
        else:
            # Per-sample positions: scatter-style. There's no native vectorized
            # form for "advanced-index assignment with a per-batch index" that
            # avoids a Python loop. For batch=1 (the common HE+ case) the loop
            # is one iteration; for larger batches it's still a tiny number.
            B = positions.shape[0]
            for b in range(B):
                self._K[layer][b, :, positions[b], :] = K[b]
                self._V[layer][b, :, positions[b], :] = V[b]

    def read_full(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full ``(K, V)`` tensors for `layer`.

        Each is shape ``[B, n_kv_heads, max_length, head_dim]``. The model
        consumes these in its attention, masking out positions that the
        engine has not yet supplied K/V for via the attention mask.
        """
        return self._K[layer], self._V[layer]

    def commit(self, positions: torch.Tensor) -> None:
        """Mark `positions` as frozen — engine should skip recomputing K/V there."""
        if positions.ndim == 1:
            self._commit_state[:, positions] = True
        elif positions.ndim == 2:
            B = positions.shape[0]
            for b in range(B):
                self._commit_state[b, positions[b]] = True
        else:
            raise ValueError(
                f"positions must be 1D ([n_pos]) or 2D ([B, n_pos]), "
                f"got {positions.ndim}D shape={tuple(positions.shape)}"
            )

    def commit_state(self) -> torch.Tensor:
        return self._commit_state

    def reset(self) -> None:
        """Cheaply mark all positions uncommitted. Tensors are NOT reallocated."""
        self._commit_state.zero_()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_positions(self, positions: torch.Tensor, K: torch.Tensor) -> None:
        """Check shape contract: positions and K must agree on n_pos."""
        if positions.ndim == 1:
            if K.shape[2] != positions.shape[0]:
                raise ValueError(
                    f"K.shape[2]={K.shape[2]} != positions.shape[0]={positions.shape[0]}"
                )
        elif positions.ndim == 2:
            if positions.shape[0] != self.batch_size:
                raise ValueError(
                    f"positions.shape[0]={positions.shape[0]} != batch_size={self.batch_size}"
                )
            if K.shape[2] != positions.shape[1]:
                raise ValueError(
                    f"K.shape[2]={K.shape[2]} != positions.shape[1]={positions.shape[1]}"
                )
        else:
            raise ValueError(
                f"positions must be 1D or 2D, got {positions.ndim}D"
            )
