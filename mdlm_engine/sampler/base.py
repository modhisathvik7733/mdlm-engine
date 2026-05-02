"""Sampler protocol: confidence-based candidate selection at masked positions.

A Sampler is a callable that, given logits at masked positions, returns
``(confidences, candidates)`` so the scheduler can decide which to commit
this step.

This is intentionally a Protocol — registered samplers are just functions,
not classes. Keeps the surface tiny.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class Sampler(Protocol):
    """Function signature for a confidence-based sampler.

    Parameters
    ----------
    logits :
        ``[N, V]`` raw logits at the N masked positions to be sampled.
    temperature :
        Standard temperature; 0 means deterministic argmax.
    top_p :
        Optional nucleus-sampling cutoff. ``None`` disables.
    top_k :
        Optional top-k filter. ``None`` disables.

    Returns
    -------
    confidences :
        ``[N]`` non-negative scores. Higher = more confident. The scheduler
        decides which positions to commit based on these.
    candidates :
        ``[N]`` long; the candidate token id for each position.
    """

    def __call__(
        self,
        logits: "torch.Tensor",
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]": ...


# Registry: name -> Sampler callable.
_SAMPLER_REGISTRY: dict[str, Sampler] = {}


def register_sampler(name: str):
    """Decorator: register a sampler under a string name."""

    def decorator(fn: Sampler) -> Sampler:
        if name in _SAMPLER_REGISTRY:
            raise ValueError(f"sampler '{name}' already registered")
        _SAMPLER_REGISTRY[name] = fn
        return fn

    return decorator


def get_sampler(name: str) -> Sampler:
    try:
        return _SAMPLER_REGISTRY[name]
    except KeyError as e:
        known = sorted(_SAMPLER_REGISTRY)
        raise KeyError(
            f"no sampler named '{name}'. Known: {known}."
        ) from e
