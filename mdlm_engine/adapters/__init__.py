"""Per-model adapters — the only place model-specific code lives."""
import importlib

from mdlm_engine.adapters.base import (
    AdapterOutput,
    CacheLayout,
    ModelAdapter,
    ValidationReport,
    get_adapter_for,
    register_adapter,
)

# Side-effect imports register adapters via @register_adapter.
for _mod in ("dream",):
    importlib.import_module(f"mdlm_engine.adapters.{_mod}")
del _mod

__all__ = [
    "ModelAdapter",
    "AdapterOutput",
    "CacheLayout",
    "ValidationReport",
    "register_adapter",
    "get_adapter_for",
]
