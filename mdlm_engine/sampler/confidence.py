"""Confidence-based samplers, ported and generalized from Dream-Coder's
``fast_dllm/generation_utils_block.py:85-123``.

Five samplers, all model-agnostic:

- **argmax** (deterministic): pick the most-likely token; confidence = its prob.
- **entropy**: pick by negentropy; confidence = -H(p) (higher = lower entropy).
- **margin**: pick top-1; confidence = top1 - top2 (sharper distribution wins).
- **maskgit_plus**: pick top-1; confidence = top-1 prob (alias for "max prob").
- **topk_margin**: top-k filter then margin scoring.

All five are registered into the sampler registry under their names. The engine
selects via ``DiffusionEngine(..., sampler="entropy")`` — name lookup goes
through ``mdlm_engine.sampler.base.get_sampler``.

These samplers are intentionally stateless and side-effect-free. They take a
``[N, V]`` logits tensor (already gathered at masked positions) and return
``(confidences, candidates)`` per the ``Sampler`` protocol.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mdlm_engine.sampler.base import register_sampler

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers — top-k / top-p filtering (run BEFORE temperature scaling, because
# we want to clamp the *raw* logits before softmax; this matches HF / Dream).
# ---------------------------------------------------------------------------


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits below the k-th highest to -inf. Returns a new tensor."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: mask the tail beyond cumulative-prob top_p."""
    if not (0.0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_to_remove = cumulative > top_p
    # Shift right so the first token above the threshold is kept.
    sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
    sorted_to_remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        -1, sorted_indices, sorted_to_remove,
    )
    return logits.masked_fill(mask, float("-inf"))


def _apply_filters(
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> torch.Tensor:
    if top_k is not None:
        logits = _top_k_filter(logits, int(top_k))
    if top_p is not None:
        logits = _top_p_filter(logits, float(top_p))
    if temperature > 0.0:
        logits = logits / temperature
    return logits


def _sample_token(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Pick one token per row.

    Returns ``[N]`` long. With temperature 0 the result is argmax;
    otherwise it's a multinomial draw.
    """
    if temperature <= 0.0:
        return probs.argmax(dim=-1)
    # multinomial requires non-negative; softmax already enforces.
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# The five samplers
# ---------------------------------------------------------------------------


@register_sampler("argmax")
def argmax_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic: pick argmax. Confidence = top-1 prob."""
    filtered = _apply_filters(logits, temperature, top_p, top_k)
    probs = F.softmax(filtered, dim=-1)
    confidences, candidates = probs.max(dim=-1)
    return confidences, candidates


@register_sampler("maskgit_plus")
def maskgit_plus_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same scoring as ``argmax`` but tokens drawn by the temperature mode.

    At T=0 this collapses to argmax. At T>0, candidates are sampled from
    the distribution and confidences are the sampled token's probability.
    """
    filtered = _apply_filters(logits, temperature, top_p, top_k)
    probs = F.softmax(filtered, dim=-1)
    candidates = _sample_token(probs, temperature)
    confidences = probs.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
    return confidences, candidates


@register_sampler("entropy")
def entropy_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence = negative Shannon entropy (- sum p log p).

    Sharper distributions score higher. Tokens are still drawn per the
    temperature mode (argmax at T=0, multinomial at T>0).
    """
    filtered = _apply_filters(logits, temperature, top_p, top_k)
    probs = F.softmax(filtered, dim=-1)
    eps = 1e-10
    neg_entropy = (probs * torch.log(probs + eps)).sum(dim=-1)  # = -H(p)
    candidates = _sample_token(probs, temperature)
    return neg_entropy, candidates


@register_sampler("margin")
def margin_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence = top1 - top2 probability gap (margin).

    Larger margin → more confident this token over its runner-up.
    """
    filtered = _apply_filters(logits, temperature, top_p, top_k)
    probs = F.softmax(filtered, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    margin = sorted_probs[..., 0] - sorted_probs[..., 1]
    candidates = _sample_token(probs, temperature)
    return margin, candidates


@register_sampler("topk_margin")
def topk_margin_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k filter, then margin scoring on the remaining mass.

    Convenient alias for ``margin`` with ``top_k`` defaulted to 5 if the
    caller didn't pass one.
    """
    if top_k is None:
        top_k = 5
    return margin_sampler(logits, temperature=temperature, top_p=top_p, top_k=top_k)
