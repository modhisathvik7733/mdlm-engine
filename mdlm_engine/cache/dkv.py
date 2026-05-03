"""DKVCache — Phase-1 headline KV-cache (delayed KV; arxiv 2505.15781).

Storage layout is the same as ``BlockCache``: pre-allocated
``[B, n_kv_heads, max_length, head_dim]`` workspace tensors per layer plus a
``[B, max_length]`` bool commit-state mask.

What makes this **dKV** rather than block-prefix:

1. **Stricter immutability.** Once a position is committed, its K/V is
   considered frozen. ``replace_at`` on a committed position raises
   ``CommittedSlotWriteError`` by default — catches engine/scheduler bugs
   that would silently corrupt the cache (and quietly drop pass@1).
   Set ``strict=False`` to downgrade to a warning.

2. **Optional one-step delay** (``one_step_delay=True``). Per the dKV paper,
   K/V emitted by the model at step t is "tentative" — it was computed with
   the position still masked. The cleaner cache flushes commits with one
   step of latency: ``commit(positions)`` queues into ``_pending_commits``,
   and the next call promotes the previous batch into ``_commit_state``.

   Default ``one_step_delay=False`` matches fast_dllm's ``dual_cache``
   semantics (commit immediately) — verified by the equivalence test in
   Phase 1 day 8. Flip the flag and re-run if drift is observed.

3. **Engine policy.** A ``DKVCache`` is intended to be used by an engine that
   only recomputes K/V at currently-masked positions (skipping committed
   ones). ``BlockCache`` is for engines that recompute the whole active block
   every step. The cache itself is policy-agnostic; the engine
   (``core/loop.py``, day 5) picks which positions to feed in via
   ``replace_at``.

Storage cost is identical to ``BlockCache`` (~512 MB at 7B/GQA8/4096-ctx/bf16
across 32 layers at batch 1). Speedup vs ``BlockCache`` comes from the engine
asking the adapter to compute K/V for fewer positions per step.
"""
from __future__ import annotations

import warnings

import torch

from mdlm_engine.cache.base import DiffusionCache


class CommittedSlotWriteError(RuntimeError):
    """Raised when ``DKVCache.replace_at`` is asked to overwrite a slot
    whose K/V has already been committed (frozen).

    This is the engine's smoke alarm: a scheduler that re-asks the adapter
    to recompute a committed position is wasting compute and risking
    numerical drift. Catch the bug here, not at hour 12 of a benchmark.
    """


class DKVCache(DiffusionCache):
    """Position-indexed delayed KV cache (dKV; arxiv 2505.15781)."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        *,
        strict: bool = True,
        one_step_delay: bool = False,
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
        self.strict = strict
        self.one_step_delay = one_step_delay

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
        self._commit_state = torch.zeros(
            batch_size, max_length, dtype=torch.bool, device=device,
        )
        # Pending commits — only used when one_step_delay=True.
        self._pending_commits = torch.zeros(
            batch_size, max_length, dtype=torch.bool, device=device,
        )

    # ------------------------------------------------------------------
    # Storage operations
    # ------------------------------------------------------------------

    def replace_at(
        self,
        layer: int,
        positions: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Write K/V into cache slots indexed by ``positions``.

        Differs from ``BlockCache.replace_at`` only in that it refuses to
        overwrite an already-committed slot (engine bug detector).
        """
        self._validate_positions(positions, K)
        self._check_no_committed_overlap(positions)

        if positions.ndim == 1:
            self._K[layer].index_copy_(2, positions, K)
            self._V[layer].index_copy_(2, positions, V)
        else:
            B = positions.shape[0]
            for b in range(B):
                self._K[layer][b, :, positions[b], :] = K[b]
                self._V[layer][b, :, positions[b], :] = V[b]

    def read_full(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._K[layer], self._V[layer]

    def commit(self, positions: torch.Tensor) -> None:
        """Mark ``positions`` as frozen.

        With ``one_step_delay=False`` (default): commits take effect immediately.
        With ``one_step_delay=True``: positions move from ``_pending_commits`` to
        ``_commit_state`` on the *next* commit call (one-step latency, matching
        the dKV paper's "delayed" semantics).
        """
        if self.one_step_delay:
            # First, promote the previous pending commits to actually committed.
            self._commit_state |= self._pending_commits
            self._pending_commits.zero_()
            # Now record the new ones as pending — they'll commit next call.
            self._mark_positions(self._pending_commits, positions)
        else:
            self._mark_positions(self._commit_state, positions)

    def commit_state(self) -> torch.Tensor:
        return self._commit_state

    def reset(self) -> None:
        """Cheap reuse: zero both commit + pending masks. K/V stays."""
        self._commit_state.zero_()
        self._pending_commits.zero_()

    def update_from_model_output(
        self,
        layer: int,
        past_key_values_layer: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
    ) -> None:
        """Write the model's returned K/V back into the cache, bypassing
        the strict committed-slot check.

        Why we override the default:
        - ``replace_at`` runs ``_check_no_committed_overlap`` to catch
          ENGINE bugs (scheduler re-asks for recompute at a committed slot).
        - But the model's own returned K/V is authoritative — by the time
          ``model.forward`` returns, the values it produced are the new
          truth even at positions that *happen* to overlap commit_state.
        - Phase 2 will primarily use the alias-based in-place write path
          (``to_legacy_kv`` returns aliases of ``_K``/``_V``; the model's
          ``dual_cache=True`` writes directly through). This method is the
          fallback for models that return a non-aliased ``past_key_values``.

        Same shape contract as ``replace_at``:
          - ``positions``: ``[n_pos]`` (broadcast) or ``[B, n_pos]``
          - K, V: ``[B, n_kv_heads, n_pos, head_dim]``
        """
        self._validate_positions(positions, past_key_values_layer[0])
        K, V = past_key_values_layer
        if positions.ndim == 1:
            self._K[layer].index_copy_(2, positions, K)
            self._V[layer].index_copy_(2, positions, V)
        else:
            B = positions.shape[0]
            for b in range(B):
                self._K[layer][b, :, positions[b], :] = K[b]
                self._V[layer][b, :, positions[b], :] = V[b]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mark_positions(
        self,
        target: torch.Tensor,  # [B, L_max] bool, modified in-place
        positions: torch.Tensor,
    ) -> None:
        if positions.ndim == 1:
            target[:, positions] = True
        elif positions.ndim == 2:
            B = positions.shape[0]
            for b in range(B):
                target[b, positions[b]] = True
        else:
            raise ValueError(
                f"positions must be 1D ([n_pos]) or 2D ([B, n_pos]), "
                f"got {positions.ndim}D shape={tuple(positions.shape)}"
            )

    def _check_no_committed_overlap(self, positions: torch.Tensor) -> None:
        """Refuse (or warn on) a write that would overwrite committed K/V."""
        # Materialize a [B, n_pos] index of the requested writes.
        if positions.ndim == 1:
            committed_at_positions = self._commit_state[:, positions]
        else:
            committed_at_positions = torch.gather(
                self._commit_state, dim=1, index=positions,
            )
        if not committed_at_positions.any():
            return

        msg = (
            f"DKVCache.replace_at asked to overwrite committed slots. "
            f"Engine bug — a committed position should not be recomputed. "
            f"Committed-overlap mask shape={tuple(committed_at_positions.shape)}, "
            f"sum={int(committed_at_positions.sum())}."
        )
        if self.strict:
            raise CommittedSlotWriteError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_positions(self, positions: torch.Tensor, K: torch.Tensor) -> None:
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
