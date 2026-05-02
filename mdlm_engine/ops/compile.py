"""torch.compile wrapper with static-shape padding.

Phase-1 strategy (from Day-1 spike #2: cpu/gpu ratio = 0.66):
``torch.compile(mode="reduce-overhead", fullgraph=False)`` is a Phase-1
*nice-to-have*, not load-bearing. Expected ~10-25% speedup.

Why static-shape padding matters: ``mode="reduce-overhead"`` uses CUDA
graphs, which require static input shapes. Diffusion sampling has dynamic
shapes (the active block shrinks as positions commit). The fix: pad the
active window to a fixed ``block_length`` and rely on the attention mask
to ignore the pad. This keeps the captured graph reusable across all
diffusion steps within a block.

For Phase 1 we expose a tiny helper ``maybe_compile_model`` that the
engine can call with the user's ``--compile`` flag. We DO NOT compile
the engine's Python loop — only the model.forward call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def maybe_compile_model(
    model: "torch.nn.Module",
    *,
    enabled: bool,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
) -> "torch.nn.Module":
    """Return ``torch.compile(model, ...)`` if enabled, else ``model``.

    Wrapped in try/except: compile failures (graph breaks, dynamic shape
    issues, Blackwell + nightly incompatibilities) fall back to the
    uncompiled model with a warning. Day-1 spike #2 informed the choice
    of ``mode="reduce-overhead"``.
    """
    if not enabled:
        return model
    try:
        import torch  # local import to keep top-level cheap

        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        return compiled  # type: ignore[no-any-return]
    except Exception as e:  # noqa: BLE001 — fallback is intentionally broad
        import warnings

        warnings.warn(
            f"torch.compile(mode={mode!r}, fullgraph={fullgraph}) failed: "
            f"{type(e).__name__}: {e}. Falling back to uncompiled model.",
            RuntimeWarning, stacklevel=2,
        )
        return model


def pad_active_window_to_block_length(
    active_ids: "torch.Tensor",
    block_length: int,
    pad_token_id: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pad ``active_ids`` (shape ``[B, n]`` with n ≤ block_length) on the
    right to exactly ``block_length`` columns, using ``pad_token_id`` as
    the fill. Returns ``(padded_ids, valid_mask)`` where ``valid_mask``
    is ``[B, block_length]`` bool, True for real positions.

    Used so CUDA graphs see a fixed shape per block. The model produces
    logits at all block_length positions; the engine ignores the padded
    ones via the valid_mask.
    """
    import torch

    B, n = active_ids.shape
    if n == block_length:
        return active_ids, torch.ones(
            B, block_length, dtype=torch.bool, device=active_ids.device,
        )
    if n > block_length:
        raise ValueError(
            f"active_ids has {n} columns, exceeds block_length={block_length}"
        )

    pad = torch.full(
        (B, block_length - n), pad_token_id,
        dtype=active_ids.dtype, device=active_ids.device,
    )
    padded = torch.cat([active_ids, pad], dim=-1)
    mask = torch.zeros(B, block_length, dtype=torch.bool, device=active_ids.device)
    mask[:, :n] = True
    return padded, mask
