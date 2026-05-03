"""DiffusionCache ABC — position-indexed K/V storage for masked diffusion LMs.

Design rationale: see plan §"DiffusionCache ABC" and §"Why not subclass HF Cache".

The two operations the standard HF `Cache` API does NOT expose well:
    1. replace_at(layer, positions, K, V): in-place overwrite at given positions
    2. read_full(layer): return full K/V tensors for the model's attention

Both `fast_dllm` and `dKV-Cache` (horseee/dkv-cache) use raw tuple-of-tensors
with in-place writes — we generalize that pattern.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DiffusionCache(ABC):
    """ABC for masked-diffusion KV caches.

    Conventions (verified against fast_dllm + dKV-Cache reference impls):

    - K/V tensor shape per layer: ``[B, n_kv_heads, L_max, head_dim]``
      (HF legacy "kv_bld" layout). Adapter declares this via `cache_layout()`.

    - Positions are absolute indices into the sequence. The cache is
      pre-allocated to `L_max` and `replace_at(...)` writes into specific slots.
      A `commit_state()` bool tensor `[B, L_max]` tracks which positions have
      "frozen" K/V the model can read; the rest must be recomputed.

    - All methods are batch-aware. `positions` is `[B, n_pos]` if positions
      differ per-sample, or `[n_pos]` if shared across the batch (broadcast).

    - The cache is allocated once per generation call. `reset()` zeroes the
      commit state without freeing memory (cheap re-use across generations).

    Subclasses implement different replacement semantics:
        - `BlockCache`: cache prefix (prompt + finalized blocks); recompute
          everything in the active block each step. Equivalent to fast_dllm's
          non-`dual_cache` path. Phase 1.
        - `DKVCache`: cache only committed (decided non-mask) tokens; recompute
          K/V only at currently-masked positions. Phase 1; arxiv 2505.15781.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_length: int,
        batch_size: int,
        dtype: "torch.dtype",
        device: "torch.device | str",
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

    # ------------------------------------------------------------------
    # Mandatory storage operations
    # ------------------------------------------------------------------

    @abstractmethod
    def replace_at(
        self,
        layer: int,
        positions: "torch.Tensor",  # [n_pos] or [B, n_pos] long
        K: "torch.Tensor",  # [B, n_kv, n_pos, d_h]
        V: "torch.Tensor",  # [B, n_kv, n_pos, d_h]
    ) -> None:
        """Write `K`, `V` into cache slots `[positions]` for `layer`.

        In-place, no return. Equivalent (modulo shape conventions) to
        ``past_key[:, :, positions, :] = K`` from fast_dllm.
        """

    @abstractmethod
    def read_full(
        self,
        layer: int,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return the full `(K, V)` tensors for `layer`.

        Shape: each `[B, n_kv_heads, L_max, head_dim]`. The model's attention
        then masks against `commit_state()` so uncommitted positions don't
        contribute (or they're recomputed; see `BlockCache` vs `DKVCache`).
        """

    @abstractmethod
    def commit(
        self,
        positions: "torch.Tensor",  # [n_pos] or [B, n_pos] long
    ) -> None:
        """Mark `positions` as having frozen K/V. The scheduler calls this
        when a token is decided to be no-mask for the rest of generation.
        """

    @abstractmethod
    def commit_state(self) -> "torch.Tensor":
        """Return `[B, L_max]` bool: True where K/V is frozen and reusable."""

    @abstractmethod
    def reset(self) -> None:
        """Zero out the commit state (and optionally clear K/V tensors).

        Tensors are not deallocated — just marked unused. Cheap to call
        between generations.
        """

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def update_from_model_output(
        self,
        layer: int,
        past_key_values_layer: "tuple[torch.Tensor, torch.Tensor]",
        positions: "torch.Tensor",
    ) -> None:
        """Bridge from `model.forward()` output back into the cache.

        Default impl assumes the model returned `(K, V)` for exactly the
        positions we asked it to compute. Subclasses can override if their
        model returns full-sequence K/V and they need to slice.
        """
        K, V = past_key_values_layer
        self.replace_at(layer, positions, K, V)

    def to_legacy_kv(self) -> "list[tuple[torch.Tensor, torch.Tensor]]":
        """Materialize the cache as HF's legacy ``past_key_values`` shape:
        a list of ``(K, V)`` per layer, each ``[B, n_kv_heads, L_max, head_dim]``.

        This is what every Dream/Llama/Qwen-family modeling forward expects when
        you pass ``past_key_values=...``. Phase 2 wires this into
        ``adapter.forward()`` so the model can reuse K/V across diffusion steps.

        Default impl assumes the cache stores `_K[layer]` / `_V[layer]` lists
        of tensors — the layout used by ``BlockCache`` and ``DKVCache``.
        Subclasses with a different storage layout (e.g. a future ``SparseCache``)
        should override.

        Raises ``NotImplementedError`` if the subclass doesn't expose ``_K``/``_V``.
        ``NoOpCache`` (the equivalence baseline) returns ``None`` for each layer
        — telling the model "no cache, recompute everything" — so it overrides.
        """
        K_list = getattr(self, "_K", None)
        V_list = getattr(self, "_V", None)
        if K_list is None or V_list is None:
            raise NotImplementedError(
                f"{type(self).__name__} doesn't expose _K/_V for "
                f"to_legacy_kv. Override the method on the subclass."
            )
        return [(K_list[i], V_list[i]) for i in range(self.n_layers)]

    def attention_mask_for_step(
        self,
        active_positions: "torch.Tensor",
    ) -> "torch.Tensor | str":
        """What attention mask should the model see this step?

        Default: ``"bidirectional"`` — model attends to everything in
        ``[0, max_length)`` it has K/V for, plus the active positions it's
        recomputing. Subclasses can return a 4D bool mask if they want to
        restrict attention to committed + active.
        """
        del active_positions  # default ignores; subclasses can use
        return "bidirectional"

    @property
    def num_committed(self) -> "torch.Tensor":
        """`[B]` long: number of committed (frozen) positions per sample.

        Useful for debugging and scheduler logic.
        """
        return self.commit_state().sum(dim=-1)


class NoOpCache(DiffusionCache):
    """Sentinel "no cache" implementation — every step recomputes everything.

    Useful as the equivalence baseline for tests: `cache="none"` should
    produce logits identical to vanilla `model(input_ids)` without any cache.

    Implementation: stores nothing, returns empty tensors, `commit_state()`
    is always all-False.
    """

    def replace_at(self, layer, positions, K, V):  # noqa: D401
        del layer, positions, K, V  # NoOp stores nothing
        return None

    def read_full(self, layer):
        del layer  # all layers return the same empty tensor
        import torch

        empty = torch.empty(
            self.batch_size, self.n_kv_heads, 0, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        return empty, empty

    def commit(self, positions):
        del positions  # NoOp tracks no commits
        return None

    def to_legacy_kv(self):
        """NoOpCache returns ``None`` per layer — signals "recompute everything"."""
        return [None] * self.n_layers

    def commit_state(self):
        import torch

        return torch.zeros(
            self.batch_size, self.max_length, dtype=torch.bool, device=self.device,
        )

    def reset(self):
        return None
