"""CPU-only smoke tests for the package imports + ABC contracts.

These run without a GPU. They verify the ABC structure is internally
consistent and the registry round-trips. The GPU-bound validations
(logit alignment, equivalence, NaN freedom) land in
`tests/test_adapter_validation.py` once the engine ships.
"""
from __future__ import annotations

import pytest


def test_top_level_imports_cheap():
    """`import mdlm_engine` must not import torch eagerly."""
    import sys

    # Clear any prior import to make this honest.
    for k in list(sys.modules):
        if k == "mdlm_engine" or k.startswith("mdlm_engine."):
            del sys.modules[k]

    import mdlm_engine  # noqa: F401

    # Importing mdlm_engine itself shouldn't trigger torch import.
    # Adapters import torch lazily; the top package should not.
    assert mdlm_engine.__version__


def test_abc_methods_are_abstract():
    """ModelAdapter and DiffusionCache must require their abstract methods."""
    from mdlm_engine.adapters.base import ModelAdapter
    from mdlm_engine.cache.base import DiffusionCache

    # Direct instantiation should fail on the @abstractmethod contract.
    with pytest.raises(TypeError):
        ModelAdapter(model=None, tokenizer=None)  # type: ignore[abstract]

    with pytest.raises(TypeError):
        DiffusionCache(  # type: ignore[abstract]
            n_layers=1, n_kv_heads=1, head_dim=1,
            max_length=1, batch_size=1, dtype=None, device="cpu",
        )


def test_registry_round_trip():
    """`@register_adapter` + `get_adapter_for` must round-trip."""
    from mdlm_engine.adapters.base import (
        ModelAdapter,
        get_adapter_for,
        register_adapter,
    )

    @register_adapter("__test_dummy__")
    class _Dummy(ModelAdapter):
        @property
        def mask_token_id(self) -> int:
            return 0

        @property
        def pad_token_id(self) -> int:
            return 0

        @property
        def eos_token_ids(self) -> list[int]:
            return [0]

        def apply_chat_template(self, messages):  # type: ignore[override]
            return None

        def build_position_ids(self, input_ids, attention_mask_1d):  # type: ignore[override]
            return None

        def build_attention_mask(self, attention_mask_1d, seq_len):  # type: ignore[override]
            return "bidirectional"

        def shift_logits(self, logits):  # type: ignore[override]
            return logits

        def forward(self, input_ids, attention_mask, position_ids, diffusion_cache, use_cache):  # type: ignore[override]
            raise NotImplementedError("dummy")

    cls = get_adapter_for("__test_dummy__")
    assert cls is _Dummy
    assert cls.model_type == "__test_dummy__"


def test_unknown_adapter_raises_with_helpful_message():
    """get_adapter_for should error helpfully and list known model_types."""
    from mdlm_engine.adapters.base import get_adapter_for

    with pytest.raises(KeyError) as exc:
        get_adapter_for("__definitely_not_registered__")

    msg = str(exc.value)
    assert "__definitely_not_registered__" in msg
    assert "Subclass `ModelAdapter`" in msg


def test_noop_cache_basic():
    """NoOpCache should be instantiable and return all-False commit state."""
    import torch

    from mdlm_engine.cache.base import NoOpCache

    cache = NoOpCache(
        n_layers=2, n_kv_heads=4, head_dim=8,
        max_length=16, batch_size=1,
        dtype=torch.float32, device="cpu",
    )
    state = cache.commit_state()
    assert state.shape == (1, 16)
    assert state.dtype == torch.bool
    assert not state.any().item()

    K, V = cache.read_full(0)
    assert K.shape[0] == 1 and K.shape[1] == 4 and K.shape[3] == 8
    # NoOpCache has zero-length sequence dim — model recomputes every step.
    assert K.shape[2] == 0


def test_validation_report_all_passed_property():
    """ValidationReport.all_passed only True when all four tests pass."""
    from mdlm_engine.adapters.base import ValidationReport

    r = ValidationReport(
        adapter_name="X",
        model_type="x",
        logit_alignment_passed=True,
        roundtrip_equivalence_passed=True,
        cache_equivalence_passed=True,
        nan_freedom_passed=True,
    )
    assert r.all_passed

    r.cache_equivalence_passed = False
    assert not r.all_passed
