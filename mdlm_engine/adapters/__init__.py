"""Per-model adapters — the only place model-specific code lives."""
from mdlm_engine.adapters.base import (
    ModelAdapter,
    AdapterOutput,
    CacheLayout,
    ValidationReport,
    register_adapter,
    get_adapter_for,
)

__all__ = [
    "ModelAdapter",
    "AdapterOutput",
    "CacheLayout",
    "ValidationReport",
    "register_adapter",
    "get_adapter_for",
]
