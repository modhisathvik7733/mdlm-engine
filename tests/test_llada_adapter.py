"""CPU-only tests for LLaDAAdapter.

Verifies that LLaDA's *different* conventions (vs Dream) are encoded
correctly: identity shift_logits, None position_ids, 2D attention mask,
custom mask token id, GQA-None handling.

Together with `test_dream_adapter.py`, these prove the model-agnostic claim
on a CPU box: same engine code paths handle both adapters cleanly.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _StubLLaDATokenizer:
    """Stub mirroring `GSAI-ML/LLaDA-8B-Base`'s tokenizer surface."""

    def __init__(self, mask_id: int = 126336):
        self._mask_id = mask_id
        self.unk_token_id = 0
        self.pad_token_id = 126081       # <|endoftext|>
        self.eos_token_id = 126081
        self._extra = {"<|endoftext|>": 126081, "<|eot_id|>": 126349}

    def convert_tokens_to_ids(self, token):
        if token == "<|mdm_mask|>":
            return self._mask_id
        return self._extra.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, return_tensors, return_dict, add_generation_prompt):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}


def _stub_llada_config(num_kv_heads=None):
    """LLaDA's published config has num_key_value_heads=None (not absent)."""
    return SimpleNamespace(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=num_kv_heads,
        head_dim=128,
        hidden_size=4096,
    )


# ---------------------------------------------------------------------------
# Registration + construction
# ---------------------------------------------------------------------------


def test_llada_registered():
    from mdlm_engine.adapters.base import get_adapter_for
    from mdlm_engine.adapters.llada import LLaDAAdapter

    assert get_adapter_for("llada") is LLaDAAdapter


def test_construction_with_correct_mask_id():
    from mdlm_engine.adapters.llada import LLADA_MASK_TOKEN_ID, LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    tok = _StubLLaDATokenizer(mask_id=LLADA_MASK_TOKEN_ID)
    adapter = LLaDAAdapter(model=model, tokenizer=tok)

    assert adapter.mask_token_id == LLADA_MASK_TOKEN_ID
    assert adapter.pad_token_id == 126081
    assert 126081 in adapter.eos_token_ids
    assert 126349 in adapter.eos_token_ids  # <|eot_id|>


def test_construction_warns_on_mismatched_mask_id():
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    tok = _StubLLaDATokenizer(mask_id=99999)
    with pytest.warns(RuntimeWarning, match="mask_token_id"):
        LLaDAAdapter(model=model, tokenizer=tok)


def test_construction_raises_when_mask_token_unknown():
    from mdlm_engine.adapters.llada import LLaDAAdapter

    class _Bad(_StubLLaDATokenizer):
        def convert_tokens_to_ids(self, token):
            return 0

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    with pytest.raises(ValueError, match="<\\|mdm_mask\\|>"):
        LLaDAAdapter(model=model, tokenizer=_Bad())


# ---------------------------------------------------------------------------
# The convention differences from Dream
# ---------------------------------------------------------------------------


def test_shift_logits_is_identity():
    """LLaDA logits already align — no shift, just identity."""
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    logits = torch.randn(2, 5, 100)
    assert torch.equal(adapter.shift_logits(logits), logits)


def test_build_position_ids_returns_none():
    """LLaDA's forward doesn't accept position_ids — adapter returns None.
    Spike confirmed: forward_accepts.position_ids = False."""
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    assert adapter.build_position_ids(torch.zeros(1, 4, dtype=torch.long), None) is None
    assert adapter.build_position_ids(
        torch.zeros(1, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 1]]),
    ) is None


def test_build_attention_mask_returns_2d_for_llada():
    """LLaDA expects a standard 2D `[B, L]` mask."""
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    attn1d = torch.tensor([[1, 1, 0, 0]])
    out = adapter.build_attention_mask(attn1d, seq_len=4)
    assert out.shape == (1, 4)  # 2D, not 4D like Dream
    assert torch.equal(out, attn1d)


def test_build_attention_mask_bidirectional_sentinel_when_no_mask():
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    assert adapter.build_attention_mask(None, seq_len=4) == "bidirectional"


# ---------------------------------------------------------------------------
# GQA-None handling — the LLaDA-specific quirk
# ---------------------------------------------------------------------------


def test_n_kv_heads_falls_back_when_config_is_none():
    """LLaDA-8B-Base's config sets num_key_value_heads=None (not absent).
    The ABC's default getattr-with-default doesn't trigger; the LLaDAAdapter
    overrides n_kv_heads to handle this."""
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(
        config=_stub_llada_config(num_kv_heads=None),
        parameters=lambda: iter([]),
    )
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())
    # Should fall back to num_attention_heads = 32
    assert adapter.n_kv_heads == 32


def test_n_kv_heads_uses_value_when_specified():
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(
        config=_stub_llada_config(num_kv_heads=8),
        parameters=lambda: iter([]),
    )
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())
    assert adapter.n_kv_heads == 8


# ---------------------------------------------------------------------------
# Cache + portability
# ---------------------------------------------------------------------------


def test_llada_uses_hf_legacy_kv_layout():
    from mdlm_engine.adapters.base import CacheLayout
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    assert adapter.cache_layout() == CacheLayout.HF_LEGACY_KV_BLD


def test_engine_can_construct_with_llada_adapter():
    """Cross-model portability smoke: same DiffusionEngine constructor
    accepts the LLaDA adapter without any engine code changes."""
    from mdlm_engine import DiffusionEngine
    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = SimpleNamespace(config=_stub_llada_config(), parameters=lambda: iter([]))
    adapter = LLaDAAdapter(model=model, tokenizer=_StubLLaDATokenizer())

    engine = DiffusionEngine(
        model, adapter=adapter,
        cache="dkv", sampler="entropy", scheduler="slowfast",
    )
    assert engine.cache_kind == "dkv"
    # n_kv_heads is the field that distinguishes models for cache allocation;
    # confirm the engine sees the right value through the adapter.
    cache = engine._build_cache(batch_size=1, max_length=64)
    assert cache.n_kv_heads == 32
    assert cache.n_layers == 32
    assert cache.head_dim == 128
