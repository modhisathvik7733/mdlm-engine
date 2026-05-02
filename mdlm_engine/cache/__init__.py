"""DiffusionCache implementations.

The engine talks only to `DiffusionCache`. Specific implementations are
selected via `DiffusionEngine(..., cache="block" | "dkv" | "none")`.

Why an internal ABC instead of subclassing `transformers.cache_utils.Cache`:
- HF's `DynamicLayer.update()` is `torch.cat([keys, key_states], dim=-2)` —
  append-only by design. Diffusion's dKV-Cache needs `replace_at(positions, K, V)`.
- Both `fast_dllm` (modeling_dream.py:484-490) and the dKV-Cache reference impl
  (github.com/horseee/dkv-cache) use raw tuple-of-tensors with in-place writes,
  not HF Cache. We generalize that pattern.
- See plan §"Why not subclass HF Cache".
"""
from mdlm_engine.cache.base import DiffusionCache, NoOpCache
from mdlm_engine.cache.block import BlockCache
from mdlm_engine.cache.dkv import CommittedSlotWriteError, DKVCache

__all__ = [
    "DiffusionCache",
    "NoOpCache",
    "BlockCache",
    "DKVCache",
    "CommittedSlotWriteError",
]
