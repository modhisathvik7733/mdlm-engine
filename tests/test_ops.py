"""CPU-only tests for ops/{compile,flexattn} + bench harness self-test."""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# compile helpers
# ---------------------------------------------------------------------------


def test_maybe_compile_disabled_returns_original_model():
    """``enabled=False`` is the no-op path."""
    from mdlm_engine.ops.compile import maybe_compile_model

    model = torch.nn.Linear(4, 4)
    out = maybe_compile_model(model, enabled=False)
    assert out is model


def test_pad_active_window_to_block_length():
    """Right-pad with pad_token_id, return correct valid mask."""
    from mdlm_engine.ops.compile import pad_active_window_to_block_length

    active = torch.tensor([[1, 2, 3]], dtype=torch.long)
    padded, mask = pad_active_window_to_block_length(
        active, block_length=8, pad_token_id=0,
    )
    assert padded.shape == (1, 8)
    assert mask.shape == (1, 8)
    # First three positions are real, last five are padding.
    assert padded[0, :3].tolist() == [1, 2, 3]
    assert (padded[0, 3:] == 0).all()
    assert mask.tolist() == [[True, True, True, False, False, False, False, False]]


def test_pad_no_op_when_already_at_block_length():
    from mdlm_engine.ops.compile import pad_active_window_to_block_length

    active = torch.zeros(1, 8, dtype=torch.long)
    padded, mask = pad_active_window_to_block_length(active, block_length=8, pad_token_id=99)
    assert padded.shape == (1, 8)
    assert mask.all().item()


def test_pad_raises_when_input_exceeds_block():
    """Defensive: a programmer error to call with too-wide input."""
    import pytest

    from mdlm_engine.ops.compile import pad_active_window_to_block_length

    active = torch.zeros(1, 9, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block_length"):
        pad_active_window_to_block_length(active, block_length=8, pad_token_id=0)


# ---------------------------------------------------------------------------
# flex_attention helpers
# ---------------------------------------------------------------------------


def test_is_flex_attention_available_returns_bool():
    """Whatever PyTorch version, the probe must not crash."""
    from mdlm_engine.ops.flexattn import is_flex_attention_available

    assert isinstance(is_flex_attention_available(), bool)


def test_bidirectional_padding_mask_mod_shape_and_logic():
    """The closure should attend iff both q_idx and kv_idx are real."""
    from mdlm_engine.ops.flexattn import bidirectional_padding_mask_mod

    attn = torch.tensor([[1, 1, 0]])  # positions 0,1 real; 2 padding
    mod = bidirectional_padding_mask_mod(attn)
    # Real-real: attend.
    assert mod(0, 0, 0, 1).item() is True
    # Real-pad: don't attend.
    assert mod(0, 0, 0, 2).item() is False
    assert mod(0, 0, 2, 0).item() is False
    # Pad-pad: don't attend.
    assert mod(0, 0, 2, 2).item() is False


# ---------------------------------------------------------------------------
# Bench harness self-test (CPU-only)
# ---------------------------------------------------------------------------


def test_bench_harness_self_test(tmp_path):
    """`--no_run` exits 0 and writes a result schema."""
    from mdlm_engine.bench.harness import main

    out = tmp_path / "result.json"
    rc = main([
        "--adapter", "dream",
        "--model_path", "stub",
        "--cache", "dkv",
        "--scheduler", "slowfast",
        "--sampler", "entropy",
        "--limit", "20",
        "--out", str(out),
        "--no_run",
    ])
    assert rc == 0
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    # Schema is intact.
    for k in [
        "adapter", "model_path", "cache", "scheduler", "sampler",
        "benchmark", "limit", "pass_at_1_single_shot", "pass_at_1_best_of_n",
        "seconds_per_problem", "tokens_per_second", "peak_vram_gb",
    ]:
        assert k in data


def test_bench_harness_diverse_configs_exposed():
    """The 8 diverse configs are importable so external eval scripts can use them."""
    from mdlm_engine.bench.harness import DIVERSE_CONFIGS

    assert len(DIVERSE_CONFIGS) == 8
    # Each config: (sampler, scheduler, temp, top_p, steps_per_block).
    for s, sch, t, tp, spb in DIVERSE_CONFIGS:
        assert isinstance(s, str) and isinstance(sch, str)
        assert 0.0 <= t <= 1.5
        assert 0.0 < tp <= 1.0
        assert spb in (16, 32)
