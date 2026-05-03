"""Cap-detection tests for DreamAdapter — does it correctly identify which
cache-wiring path a given model.forward signature supports?

PATH A: fast_dllm-patched (past_key_values + dual_cache + replace_position)
PATH B: standard HF caching only (past_key_values + use_cache)
PATH C: no cache support — fall back to v0.1.0 behavior

CPU-only; no model weights loaded.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _make_model(forward_fn):
    """Wrap a forward function as a stub model with the right config."""
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, head_dim=16, hidden_size=64,
        ),
        parameters=lambda: iter([]),
        forward=forward_fn.__get__(SimpleNamespace()),
    )
    return model


class _StubTok:
    def __init__(self):
        self.unk_token_id = 0
        self.pad_token_id = 151643
        self.eos_token_id = 151645

    def convert_tokens_to_ids(self, token):
        return {"<|mask|>": 151666, "<|im_end|>": 151645,
                "<|endoftext|>": 151643}.get(token, self.unk_token_id)


def test_caps_detect_path_a_full_fast_dllm():
    """A forward signature with all three extension kwargs → PATH A."""
    from mdlm_engine.adapters.dream import _inspect_dream_caps

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None,
                dual_cache=False, replace_position=None):
        raise NotImplementedError

    caps = _inspect_dream_caps(_make_model(forward))
    assert caps.accepts_past_key_values
    assert caps.accepts_use_cache
    assert caps.accepts_dual_cache
    assert caps.accepts_replace_position
    assert caps.path == "A"


def test_caps_detect_path_b_stock_hf_only():
    """Stock HF: past_key_values + use_cache, but no fast_dllm extensions → PATH B."""
    from mdlm_engine.adapters.dream import _inspect_dream_caps

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None):
        raise NotImplementedError

    caps = _inspect_dream_caps(_make_model(forward))
    assert caps.accepts_past_key_values
    assert caps.accepts_use_cache
    assert not caps.accepts_dual_cache
    assert not caps.accepts_replace_position
    assert caps.path == "B"


def test_caps_detect_path_c_no_caching():
    """Bare-bones forward — no caching kwargs → PATH C."""
    from mdlm_engine.adapters.dream import _inspect_dream_caps

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        raise NotImplementedError

    caps = _inspect_dream_caps(_make_model(forward))
    assert not caps.accepts_past_key_values
    assert caps.path == "C"


def test_caps_detect_path_c_partial_extensions_without_pkv():
    """Edge: dual_cache present but past_key_values missing → still PATH C.

    The path property requires past_key_values for both A and B; without it,
    fast_dllm's in-place replace pattern can't work.
    """
    from mdlm_engine.adapters.dream import _inspect_dream_caps

    def forward(self, input_ids=None, dual_cache=False, replace_position=None):
        raise NotImplementedError

    caps = _inspect_dream_caps(_make_model(forward))
    assert caps.accepts_dual_cache
    assert not caps.accepts_past_key_values
    assert caps.path == "C"


def test_caps_handle_model_without_inspectable_forward():
    """Some compiled/quantized models raise on inspect.signature → PATH C."""
    from mdlm_engine.adapters.dream import _inspect_dream_caps

    class _Weird:
        # No forward attribute at all — signature() raises AttributeError.
        pass

    caps = _inspect_dream_caps(_Weird())
    assert caps.path == "C"
    assert not caps.accepts_past_key_values


def test_dream_adapter_warns_when_no_pkv_support():
    """A model on PATH C should produce a RuntimeWarning at adapter construction
    so the user knows they're getting v0.1.0 speed."""
    import warnings

    from mdlm_engine.adapters.dream import DreamAdapter

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        raise NotImplementedError

    model = _make_model(forward)
    with pytest.warns(RuntimeWarning, match="past_key_values"):
        DreamAdapter(model=model, tokenizer=_StubTok())


def test_dream_adapter_no_warning_on_path_a():
    """A fully-fast_dllm-patched model should NOT trigger the cap warning."""
    import warnings

    from mdlm_engine.adapters.dream import DreamAdapter

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None,
                dual_cache=False, replace_position=None):
        raise NotImplementedError

    model = _make_model(forward)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Should NOT raise — no RuntimeWarning at construction.
        adapter = DreamAdapter(model=model, tokenizer=_StubTok())
    assert adapter._caps.path == "A"


def test_dream_adapter_caps_attached_to_instance():
    """Caps record is cached on the adapter for the engine to peek at."""
    from mdlm_engine.adapters.dream import DreamAdapter, _DreamCaps

    def forward(self, input_ids=None, past_key_values=None, use_cache=None):
        raise NotImplementedError

    adapter = DreamAdapter(model=_make_model(forward), tokenizer=_StubTok())
    assert isinstance(adapter._caps, _DreamCaps)
    assert adapter._caps.path == "B"
