"""mdlm-engine — model-agnostic Python inference for masked diffusion LLMs.

Public API:
    DiffusionEngine         — entrypoint, model-agnostic
    ModelAdapter            — ABC; subclass per model family
    DiffusionCache          — cache ABC (BlockCache, DKVCache implementations)
    register_adapter        — decorator to register a new adapter
    ValidationReport        — output of `adapter.validate(...)` / pytest contract

The engine talks to models exclusively through `ModelAdapter`, never through
specific model classes. Adding support for a new diffusion LM = subclass
`ModelAdapter`, implement six methods, register, pass the four-test
validation contract.

See README.md for the headline pitch and roadmap.
See `/Users/chintu/.claude/plans/jazzy-tickling-brook.md` for the full plan.
"""
from __future__ import annotations

__version__ = "0.0.1.dev0"

# Phase 1 public API. Imports kept minimal here so `import mdlm_engine` is cheap;
# the heavy modules (engine, cache, samplers) are imported lazily on first use.
from mdlm_engine.adapters.base import (  # noqa: F401
    ModelAdapter,
    AdapterOutput,
    CacheLayout,
    ValidationReport,
    register_adapter,
    get_adapter_for,
)
from mdlm_engine.cache.base import DiffusionCache  # noqa: F401
from mdlm_engine.core.engine import DiffusionEngine, GenerateOutput  # noqa: F401

__all__ = [
    "__version__",
    "ModelAdapter",
    "AdapterOutput",
    "CacheLayout",
    "ValidationReport",
    "register_adapter",
    "get_adapter_for",
    "DiffusionCache",
    "DiffusionEngine",
    "GenerateOutput",
]
