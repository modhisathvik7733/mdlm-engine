"""Generic torch helpers used by adapters and the engine.

Public:
    maybe_compile_model        torch.compile wrapper with fallback
    pad_active_window_to_block_length
    is_flex_attention_available
    bidirectional_padding_mask_mod
    flex_attention_or_sdpa
"""
from mdlm_engine.ops.compile import (
    maybe_compile_model,
    pad_active_window_to_block_length,
)
from mdlm_engine.ops.flexattn import (
    bidirectional_padding_mask_mod,
    flex_attention_or_sdpa,
    is_flex_attention_available,
)

__all__ = [
    "maybe_compile_model",
    "pad_active_window_to_block_length",
    "is_flex_attention_available",
    "bidirectional_padding_mask_mod",
    "flex_attention_or_sdpa",
]
