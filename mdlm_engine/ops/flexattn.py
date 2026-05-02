"""FlexAttention bidirectional mask helper for masked diffusion LMs.

Background (from research phase):
- ``FlashAttention-3`` does NOT support sm_120 (Blackwell). Verified via
  Dao-AILab/flash-attention#1987.
- ``FA4`` ships in nightly but is unstable on consumer Blackwell.
- ``torch.nn.functional.scaled_dot_product_attention`` falls back to a slow
  math kernel when given a 4D bool mask on Blackwell.
- ``torch.nn.attention.flex_attention`` accepts arbitrary ``mask_mod``
  closures, runs at ~90% of FA2 perf, and works on Blackwell.

For Phase 1 we expose a single helper that returns a `mask_mod` for
**bidirectional + padding** attention — exactly what masked diffusion LMs
need. Adapters can opt in by calling FlexAttention themselves inside
``forward()``; the engine doesn't depend on this path.

This is intentionally OPT-IN. Day-7 ships the helper; wiring it into a
specific adapter happens in Phase 1.5 / Phase 2 if benchmarks show the
SDPA fallback is too slow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


def is_flex_attention_available() -> bool:
    """True if ``torch.nn.attention.flex_attention`` is importable.

    FlexAttention shipped in PyTorch 2.5+. Older versions of PyTorch
    (2.4 and earlier) return False.
    """
    try:
        from torch.nn.attention import flex_attention  # noqa: F401
        return True
    except ImportError:
        return False


def bidirectional_padding_mask_mod(
    attention_mask_1d: "torch.Tensor",
) -> Callable:
    """Return a ``mask_mod`` closure for bidirectional + padding attention.

    Diffusion LMs attend to all real (non-pad) tokens regardless of
    position. Equivalent to:
        ``mask_mod(b, h, q_idx, kv_idx) = attention_mask_1d[b, q_idx] & attention_mask_1d[b, kv_idx]``
    i.e. token at position q_idx attends to token at position kv_idx iff
    both are real (mask=1).

    Parameters
    ----------
    attention_mask_1d :
        ``[B, L]`` bool or long. 1/True = real token; 0/False = padding.

    Returns
    -------
    A callable ``(b, h, q_idx, kv_idx) -> bool`` suitable for
    ``flex_attention(..., block_mask=create_block_mask(mask_mod, ...))``.

    Note
    ----
    ``flex_attention`` requires the mask_mod to be a pure function of its
    args; we capture ``attention_mask_1d`` via closure. Tensors used inside
    the closure must live on the same device as the query/key/value
    tensors at flex_attention call time.
    """

    am = attention_mask_1d.bool() if attention_mask_1d.dtype != bool else attention_mask_1d

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 — h unused (no per-head masking)
        del h
        return am[b, q_idx] & am[b, kv_idx]

    return mask_mod


def flex_attention_or_sdpa(*args, **kwargs):
    """Dispatch to ``flex_attention`` if available, else fall back to SDPA.

    Phase-1 Helper to keep adapter code device-agnostic. Adapters that
    want bidirectional masking on Blackwell call this; on Ampere/Hopper
    it transparently uses SDPA.
    """
    if is_flex_attention_available():
        from torch.nn.attention.flex_attention import flex_attention

        return flex_attention(*args, **kwargs)
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(*args, **kwargs)
