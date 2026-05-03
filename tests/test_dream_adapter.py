"""CPU-only tests for DreamAdapter's pure-tensor methods.

The model isn't loaded; we stub a minimal config and tokenizer so we can
exercise shift_logits / build_position_ids / build_attention_mask without
14 GB of weights. Real round-trip equivalence ships at day 8 acceptance.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Tokenizer stub
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def __init__(self, mask_id: int = 151666):
        self._mask_id = mask_id
        self.unk_token_id = 0
        self.pad_token_id = 151643  # Dream's actual pad
        self.eos_token_id = 151645
        self._extra = {"<|im_end|>": 151645, "<|endoftext|>": 151643}

    def convert_tokens_to_ids(self, token):
        if token == "<|mask|>":
            return self._mask_id
        return self._extra.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, return_tensors, return_dict, add_generation_prompt):
        # Phase 1 stub: just return prompt ids of fixed shape for the test.
        return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}


def _stub_model_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_size=64,
    )


def _stub_dream_model():
    """Stub model whose `forward` signature matches a fast_dllm-patched
    Dream-Coder (PATH A: accepts past_key_values, dual_cache, replace_position).

    Doesn't actually execute — just satisfies the cap-detection introspection.
    Use this for tests that don't care about the warning (most of them).
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        dual_cache=False,
        replace_position=None,
    ):
        raise NotImplementedError("stub")

    model = SimpleNamespace(
        config=_stub_model_config(),
        parameters=lambda: iter([]),
        forward=forward.__get__(SimpleNamespace()),  # bind as method
    )
    return model


# ---------------------------------------------------------------------------
# Construction + properties
# ---------------------------------------------------------------------------


def test_dream_adapter_registered():
    from mdlm_engine.adapters.base import get_adapter_for
    from mdlm_engine.adapters.dream import DreamAdapter

    assert get_adapter_for("dream") is DreamAdapter


def test_construction_with_correct_mask_id():
    from mdlm_engine.adapters.dream import DREAM_MASK_TOKEN_ID, DreamAdapter

    model = _stub_dream_model()
    tok = _StubTokenizer(mask_id=DREAM_MASK_TOKEN_ID)
    adapter = DreamAdapter(model=model, tokenizer=tok)

    assert adapter.mask_token_id == DREAM_MASK_TOKEN_ID
    assert adapter.pad_token_id == 151643
    assert 151645 in adapter.eos_token_ids
    assert 151643 in adapter.eos_token_ids  # both eos and im_end registered


def test_construction_warns_on_mismatched_mask_id():
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    tok = _StubTokenizer(mask_id=99999)  # NOT the canonical 151666
    with pytest.warns(RuntimeWarning, match="mask_token_id"):
        adapter = DreamAdapter(model=model, tokenizer=tok)
    assert adapter.mask_token_id == 99999  # still works, just warned


def test_construction_raises_when_mask_token_unknown():
    """If the tokenizer doesn't know <|mask|> at all, raise — not a soft warn."""
    from mdlm_engine.adapters.dream import DreamAdapter

    class _BadTok(_StubTokenizer):
        def convert_tokens_to_ids(self, token):
            return 0  # everything maps to UNK

    model = _stub_dream_model()
    with pytest.raises(ValueError, match="<\\|mask\\|>"):
        DreamAdapter(model=model, tokenizer=_BadTok())


# ---------------------------------------------------------------------------
# shift_logits — the conventions check
# ---------------------------------------------------------------------------


def test_shift_logits_is_right_shift_by_one():
    """Dream's shift: logits[:, i, :] in the OUTPUT corresponds to position
    i-1 of the INPUT (i.e., logits[:, 1, :] = original logits[:, 0, :]).
    Position 0 stays the same."""
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    L, V = 5, 10
    logits = torch.arange(L * V).reshape(1, L, V).float()
    shifted = adapter.shift_logits(logits)
    # Position 0 should equal the original position 0 (stays).
    assert torch.equal(shifted[0, 0], logits[0, 0])
    # Position 1 should equal the original position 0 (shifted right).
    assert torch.equal(shifted[0, 1], logits[0, 0])
    # Position L-1 should equal the original position L-2.
    assert torch.equal(shifted[0, L - 1], logits[0, L - 2])
    # Last logit (original position L-1) drops off.


# ---------------------------------------------------------------------------
# build_position_ids — Dream's cumsum(attn_mask) - 1 convention
# ---------------------------------------------------------------------------


def test_build_position_ids_returns_cumsum_minus_one():
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    # 1=real, 0=padding. Padding on the left: positions should be clamped at 0.
    attn = torch.tensor([[1, 1, 1, 1]])
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    pos = adapter.build_position_ids(input_ids, attn)
    assert pos.tolist() == [[0, 1, 2, 3]]

    # With trailing padding (rare but should still work).
    attn2 = torch.tensor([[1, 1, 1, 0]])
    pos2 = adapter.build_position_ids(input_ids, attn2)
    assert pos2.tolist() == [[0, 1, 2, 2]]  # last is unchanged at 2


def test_build_position_ids_returns_none_when_no_attn_mask():
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    assert adapter.build_position_ids(torch.zeros(1, 4, dtype=torch.long), None) is None


# ---------------------------------------------------------------------------
# build_attention_mask — 4D bidirectional
# ---------------------------------------------------------------------------


def test_build_attention_mask_is_4d_bidirectional():
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    # 1 real + 1 padding; expect a 2x2 bool mask: [[True, False], [False, False]].
    attn = torch.tensor([[1, 0]])
    mask = adapter.build_attention_mask(attn, seq_len=2)
    assert mask.shape == (1, 1, 2, 2)
    assert mask.dtype == torch.bool
    expected = torch.tensor([[[[True, False], [False, False]]]])
    assert torch.equal(mask, expected)


def test_build_attention_mask_returns_sentinel_when_no_attn_mask():
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    assert adapter.build_attention_mask(None, seq_len=4) == "bidirectional"


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def test_dream_uses_hf_legacy_kv_layout():
    from mdlm_engine.adapters.base import CacheLayout
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    assert adapter.cache_layout() == CacheLayout.HF_LEGACY_KV_BLD
    # Reflect the (stubbed) config.
    assert adapter.n_layers == 2
    assert adapter.n_kv_heads == 4
    assert adapter.head_dim == 16


# ---------------------------------------------------------------------------
# DiffusionEngine smoke (CPU; no real model)
# ---------------------------------------------------------------------------


def test_engine_construction_picks_named_pieces():
    """DiffusionEngine should resolve cache/sampler/scheduler by name without
    crashing. No actual generation — that needs a real model on a GPU."""
    from mdlm_engine import DiffusionEngine
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())

    engine = DiffusionEngine(
        model, adapter=adapter,
        cache="dkv", sampler="entropy", scheduler="slowfast",
    )
    assert engine.cache_kind == "dkv"
    assert callable(engine.sampler_fn)
    assert callable(engine.scheduler_fn)


def test_engine_validates_max_new_tokens_divides_block_length():
    from mdlm_engine import DiffusionEngine
    from mdlm_engine.adapters.dream import DreamAdapter

    model = _stub_dream_model()
    adapter = DreamAdapter(model=model, tokenizer=_StubTokenizer())
    engine = DiffusionEngine(model, adapter=adapter)

    with pytest.raises(ValueError, match="multiple of"):
        engine.generate(
            torch.zeros(1, 4, dtype=torch.long),
            max_new_tokens=63,    # not a multiple of 32
            block_length=32,
        )
