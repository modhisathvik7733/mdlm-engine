"""GenerationState — the mutable state threaded through the per-block loop.

Held outside the loop function so we can serialize/inspect for debugging
and so the loop itself remains a pure transformation:
    state, cache → state', cache'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class GenerationState:
    """Per-call mutable state for `DiffusionEngine.generate`.

    Attributes
    ----------
    x :
        ``[B, L_max]`` long. Holds the prompt + (gradually-revealed) generation.
        Masked positions in the active block contain ``mask_token_id``.
    attn_mask_1d :
        ``[B, L_max]`` long; 1 = real token, 0 = padding (NB: Dream's
        adapter overrides to 1 everywhere internally — we keep the standard
        HF convention here in the engine's mental model).
    block_start :
        Inclusive left edge of the active block (in absolute positions).
    block_end :
        Exclusive right edge.
    eos_seen :
        ``[B]`` bool — set True when the sample has emitted EOS in any block.
        The engine stops generating new blocks once all samples are flagged.
    history :
        Optional debugging trace. Populated when `return_trace=True` is
        passed to `generate`. List of (step, block, x_snapshot) tuples.
    """

    x: "torch.Tensor"
    attn_mask_1d: "torch.Tensor"
    block_start: int
    block_end: int
    eos_seen: "torch.Tensor"
    history: list = field(default_factory=list)
