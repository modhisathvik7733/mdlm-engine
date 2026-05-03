"""torch.compile wrapper.

v0.2.3 day-1 test (commit ``bd8e560``) showed ``mode="reduce-overhead"``
is fundamentally incompatible with PATH A: fast_dllm's
``past_key[:, replace_indices] = key_states`` mutates an input tensor,
which CUDA graphs forbid. Inductor logged
``skipping cudagraphs due to mutated inputs`` repeatedly, so we paid the
compile overhead with no graph reuse → 4x SLOWER than uncompiled.

v0.3.0 fix:
- Default mode is now ``"default"`` (Triton kernel fusion, no CUDA graphs).
- Set ``torch._inductor.config.cudagraph_support_input_mutation = True``
  before compile so dynamo doesn't bail on the in-place K/V replace
  pattern (PyTorch ≥ 2.10 supports this flag; the user's stack is 2.11
  cu130 which has it).
- ``mode="reduce-overhead"`` (CUDA graphs) is still selectable via the
  ``mode`` kwarg if a future workload runs without in-place mutations.

Expected speedup at v0.3.0 defaults: 1.05–1.15× from kernel fusion alone.
The bigger speed wins for masked diffusion are MXFP8 quantization and
self-speculative decoding (handled in their own modules); compile is a
modest free win once the cudagraph bust is fixed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def maybe_compile_model(
    model: "torch.nn.Module",
    *,
    enabled: bool,
    mode: str = "default",
    fullgraph: bool = False,
) -> "torch.nn.Module":
    """Return ``torch.compile(model, ...)`` if enabled, else ``model``.

    Wrapped in try/except: compile failures (graph breaks, dynamic shape
    issues, Blackwell + nightly incompatibilities) fall back to the
    uncompiled model with a warning.

    The default mode is ``"default"`` (no CUDA graphs) because PATH A's
    fast_dllm in-place K/V replace at ``modeling_dream.py:487`` is
    incompatible with CUDA graph capture (verified day-1 v0.2.3 test).
    Pass ``mode="reduce-overhead"`` only if the workload has no in-place
    input mutations.

    For ``mode="reduce-overhead"`` we additionally enable
    ``cudagraph_support_input_mutation`` so dynamo doesn't bail at the
    first mutated input — the model still has to opt out of using CUDA
    graphs at the offending op, but the rest of the forward gets graph
    benefits.
    """
    if not enabled:
        return model
    try:
        import torch  # local import to keep top-level cheap

        # Best-effort: enable mutation tolerance on stacks that support it.
        # PyTorch ≥ 2.10 has this flag; older versions silently ignore.
        try:
            import torch._inductor.config as _inductor_cfg
            _inductor_cfg.cudagraph_support_input_mutation = True
        except (ImportError, AttributeError):
            pass

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
