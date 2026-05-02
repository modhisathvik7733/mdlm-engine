"""Step / commit scheduler protocol.

A Scheduler decides which masked positions to commit (= un-mask) at the
current diffusion step, given the per-position confidences from the sampler.

Stateless — like ``Sampler``, schedulers are just registered functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class Scheduler(Protocol):
    """Function signature for a step scheduler.

    Parameters
    ----------
    confidences :
        Flat ``[N]`` confidences for the N currently-masked positions in the
        active block. Higher = more likely to commit.
    mask_index :
        ``[B, block_len]`` bool — True where positions are still masked
        (and therefore eligible to commit). Used by some schedulers to
        compute per-sample budgets.
    step :
        0-indexed step within the active block (``[0, steps_per_block)``).
    steps_per_block :
        Total denoising steps allocated per block (e.g., 32 for full quality,
        16 for medium, 4 for fast).
    threshold :
        Confidence cutoff used by ``confidence`` and ``slowfast`` schedulers.
        Ignored by ``uniform``.

    Returns
    -------
    commit_mask :
        ``[N]`` bool, same shape as ``confidences``. True = commit at this
        position this step. The engine writes the candidate token at these
        positions and calls ``cache.commit(positions)``.
    """

    def __call__(
        self,
        confidences: "torch.Tensor",
        mask_index: "torch.Tensor",
        step: int,
        steps_per_block: int,
        threshold: float = 0.9,
    ) -> "torch.Tensor": ...


_SCHEDULER_REGISTRY: dict[str, Scheduler] = {}


def register_scheduler(name: str):
    def decorator(fn: Scheduler) -> Scheduler:
        if name in _SCHEDULER_REGISTRY:
            raise ValueError(f"scheduler '{name}' already registered")
        _SCHEDULER_REGISTRY[name] = fn
        return fn

    return decorator


def get_scheduler(name: str) -> Scheduler:
    try:
        return _SCHEDULER_REGISTRY[name]
    except KeyError as e:
        known = sorted(_SCHEDULER_REGISTRY)
        raise KeyError(f"no scheduler named '{name}'. Known: {known}.") from e
