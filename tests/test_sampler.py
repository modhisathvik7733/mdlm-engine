"""Unit tests for samplers (CPU-only)."""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_five_samplers_registered():
    from mdlm_engine.sampler import get_sampler

    for name in ["argmax", "maskgit_plus", "entropy", "margin", "topk_margin"]:
        s = get_sampler(name)
        assert callable(s)


def test_unknown_sampler_raises():
    from mdlm_engine.sampler import get_sampler

    with pytest.raises(KeyError, match="argmax"):
        # Error message should include the list of known names.
        get_sampler("__nope__")


def test_duplicate_registration_raises():
    from mdlm_engine.sampler import register_sampler

    @register_sampler("__dup_test__")
    def _a(logits, temperature=0.0, top_p=None, top_k=None):
        return logits.max(dim=-1).values, logits.argmax(dim=-1)

    with pytest.raises(ValueError, match="already registered"):

        @register_sampler("__dup_test__")
        def _b(logits, temperature=0.0, top_p=None, top_k=None):
            return logits.max(dim=-1).values, logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Argmax (deterministic)
# ---------------------------------------------------------------------------


def test_argmax_picks_highest_logit():
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("argmax")
    # 2 positions, 5-token vocab. Position 0 should pick id 3, position 1 id 0.
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 5.0, 0.4],
            [3.0, 0.1, 0.2, 0.3, 0.4],
        ]
    )
    confidences, candidates = sampler(logits, temperature=0.0)
    assert candidates.tolist() == [3, 0]
    # Confidence = softmax prob of picked token (highest in each row → > 0.5).
    assert (confidences > 0.5).all()


def test_argmax_is_deterministic():
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("argmax")
    logits = torch.randn(8, 100)
    a = sampler(logits, temperature=0.0)
    b = sampler(logits, temperature=0.0)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# Margin
# ---------------------------------------------------------------------------


def test_margin_higher_when_distribution_sharper():
    """Sharper distribution (top1 >> top2) should give larger margin."""
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("margin")
    # Row 0: very sharp peak. Row 1: nearly uniform.
    sharp = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    flat = torch.tensor([[1.01, 1.0, 0.99, 0.98]])
    sharp_conf, _ = sampler(sharp, temperature=0.0)
    flat_conf, _ = sampler(flat, temperature=0.0)
    assert sharp_conf.item() > flat_conf.item()


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_entropy_is_negative_entropy():
    """Entropy sampler returns negative Shannon entropy. Higher = sharper."""
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("entropy")
    sharp = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    flat = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # uniform
    sharp_conf, _ = sampler(sharp, temperature=0.0)
    flat_conf, _ = sampler(flat, temperature=0.0)
    # Sharp dist has lower H, so -H is higher.
    assert sharp_conf.item() > flat_conf.item()
    # Uniform 4-way has H = log(4); -H ≈ -1.386
    assert flat_conf.item() == pytest.approx(-torch.log(torch.tensor(4.0)).item(), abs=1e-3)


# ---------------------------------------------------------------------------
# top-k / top-p filters
# ---------------------------------------------------------------------------


def test_top_k_filter_zeros_out_below_threshold():
    """top_k=2 should restrict picks to the top-2 logits."""
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("argmax")
    # The lower-prob choice (3.0) is well below top-1 (10.0), but top-2 (5.0)
    # should be kept. Argmax should still pick top-1.
    logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, 0.5]])
    _, cand = sampler(logits, temperature=0.0, top_k=2)
    assert cand.item() == 0


def test_top_p_filter_doesnt_break_at_t0():
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("argmax")
    logits = torch.tensor([[5.0, 4.0, 0.1, 0.1, 0.1]])
    _, cand = sampler(logits, temperature=0.0, top_p=0.9)
    assert cand.item() == 0


# ---------------------------------------------------------------------------
# topk_margin defaults
# ---------------------------------------------------------------------------


def test_topk_margin_defaults_topk_5_when_unspecified():
    """topk_margin with no top_k should still run (defaults to 5)."""
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler("topk_margin")
    logits = torch.randn(2, 100)
    confidences, candidates = sampler(logits, temperature=0.0)
    assert confidences.shape == (2,)
    assert candidates.shape == (2,)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["argmax", "maskgit_plus", "entropy", "margin", "topk_margin"])
def test_sampler_shapes_match_n_positions(name):
    """All samplers return ([N], [N]) for a [N, V] input."""
    from mdlm_engine.sampler import get_sampler

    sampler = get_sampler(name)
    logits = torch.randn(7, 1000)
    confidences, candidates = sampler(logits, temperature=0.0)
    assert confidences.shape == (7,)
    assert candidates.shape == (7,)
    assert candidates.dtype == torch.long
