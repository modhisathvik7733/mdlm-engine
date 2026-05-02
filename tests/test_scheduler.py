"""Unit tests for schedulers (CPU-only)."""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_three_schedulers_registered():
    from mdlm_engine.scheduler import get_scheduler

    for name in ["uniform", "confidence", "slowfast"]:
        s = get_scheduler(name)
        assert callable(s)


def test_unknown_scheduler_raises():
    from mdlm_engine.scheduler import get_scheduler

    with pytest.raises(KeyError):
        get_scheduler("__nope__")


# ---------------------------------------------------------------------------
# Uniform — mathematical properties
# ---------------------------------------------------------------------------


def test_uniform_progresses_at_least_one_per_step():
    """Uniform commits at least one position when there's any masked work."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("uniform")
    # 1 sample, 4 masked positions. At step 0 of 8, expect ~1 commit.
    confidences = torch.tensor([0.1, 0.2, 0.3, 0.4])
    mask_index = torch.tensor([[True, True, True, True]])
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8)
    assert commit_mask.sum() >= 1
    # Highest confidence should be among committed.
    assert commit_mask[3].item() is True


def test_uniform_pace_increases_as_steps_run_out():
    """Last step must commit ALL remaining masked positions."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("uniform")
    confidences = torch.tensor([0.1, 0.5, 0.9])
    mask_index = torch.tensor([[True, True, True]])
    # At step=7 of 8, steps_left = 1, so we must commit all 3 remaining.
    commit_mask = sched(confidences, mask_index, step=7, steps_per_block=8)
    assert commit_mask.all()


def test_uniform_zero_masked_returns_empty():
    """No work to do = no commits."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("uniform")
    commit_mask = sched(
        confidences=torch.tensor([], dtype=torch.float),
        mask_index=torch.zeros(1, 4, dtype=torch.bool),
        step=0, steps_per_block=8,
    )
    assert commit_mask.numel() == 0


def test_uniform_per_sample_correctness():
    """Two samples in a batch get independent budgets."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("uniform")
    # Sample 0: 2 masked. Sample 1: 4 masked. step=0, steps=8.
    # Expect ceil(2/8)=1 commit for sample 0, ceil(4/8)=1 for sample 1.
    confidences = torch.tensor([0.5, 0.9, 0.1, 0.2, 0.3, 0.4])
    mask_index = torch.tensor([[True, True, False, False],
                               [True, True, True, True]])
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8)
    # Sample 0's confidences are at indices 0,1 (mapping to mask_index row 0).
    # Highest is index 1 (0.9). Sample 1's highest is index 5 (0.4).
    assert commit_mask[:2].sum().item() == 1
    assert commit_mask[2:].sum().item() == 1


# ---------------------------------------------------------------------------
# Confidence-threshold — properties
# ---------------------------------------------------------------------------


def test_confidence_commits_above_threshold():
    """All positions whose confidence ≥ threshold should commit."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("confidence")
    confidences = torch.tensor([0.95, 0.5, 0.99, 0.1])
    mask_index = torch.tensor([[True, True, True, True]])
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8, threshold=0.9)
    assert commit_mask.tolist() == [True, False, True, False]


def test_confidence_falls_back_to_argmax_when_none_pass():
    """If no position passes the threshold, commit the most-confident one
    (so the diffusion loop always makes progress)."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("confidence")
    confidences = torch.tensor([0.1, 0.4, 0.2, 0.3])  # none ≥ 0.9
    mask_index = torch.tensor([[True, True, True, True]])
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8, threshold=0.9)
    assert commit_mask.sum() == 1
    assert commit_mask[1].item() is True   # argmax was at index 1 (0.4)


def test_confidence_per_sample_fallback():
    """Two-sample batch: one passes, one needs fallback."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("confidence")
    # Sample 0: nothing passes 0.9 (max=0.5, fallback to argmax)
    # Sample 1: two pass.
    confidences = torch.tensor([0.5, 0.4, 0.95, 0.92])
    mask_index = torch.tensor([[True, True, False, False],
                               [True, True, False, False]])
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8, threshold=0.9)
    # Sample 0 contributes one commit (the argmax, index 0 of its slice).
    assert commit_mask[:2].sum() == 1
    assert commit_mask[0].item() is True
    # Sample 1 contributes 2 (both pass).
    assert commit_mask[2:].sum() == 2


# ---------------------------------------------------------------------------
# SlowFast — phase transition
# ---------------------------------------------------------------------------


def test_slowfast_uses_uniform_pace_in_slow_phase_when_threshold_unmet():
    """In slow phase with no positions passing threshold, fall back to argmax."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("slowfast")
    confidences = torch.tensor([0.1, 0.4, 0.2, 0.3])
    mask_index = torch.tensor([[True, True, True, True]])
    # step=0 of 8, slow_fraction=0.5 → still in slow phase
    commit_mask = sched(confidences, mask_index, step=0, steps_per_block=8, threshold=0.9)
    # Fallback: commit at most 1 (the argmax — index 1).
    assert commit_mask.sum() == 1
    assert commit_mask[1].item() is True


def test_slowfast_dumps_aggressively_in_fast_phase():
    """In fast phase (step > slow_fraction × steps_per_block), commit
    everything ≥ threshold; if nothing passes, commit ceil(rem/steps_left)."""
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("slowfast")
    # Confidences: three pass 0.9, one doesn't.
    confidences = torch.tensor([0.95, 0.92, 0.50, 0.99])
    mask_index = torch.tensor([[True, True, True, True]])
    # step=5 of 8, slow_fraction=0.5 → step >= 4 → in fast phase
    commit_mask = sched(confidences, mask_index, step=5, steps_per_block=8, threshold=0.9)
    # All three passing should commit; the failing one should not.
    assert commit_mask.tolist() == [True, True, False, True]


def test_slowfast_zero_masked_returns_empty():
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler("slowfast")
    commit_mask = sched(
        confidences=torch.tensor([], dtype=torch.float),
        mask_index=torch.zeros(1, 4, dtype=torch.bool),
        step=0, steps_per_block=8,
    )
    assert commit_mask.numel() == 0


# ---------------------------------------------------------------------------
# Shape preservation across all three
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["uniform", "confidence", "slowfast"])
def test_scheduler_returns_bool_mask_matching_n_confidences(name):
    from mdlm_engine.scheduler import get_scheduler

    sched = get_scheduler(name)
    confidences = torch.rand(5)
    mask_index = torch.tensor([[True, True, True, True, True, False]])
    commit_mask = sched(confidences, mask_index, step=2, steps_per_block=8)
    assert commit_mask.shape == confidences.shape
    assert commit_mask.dtype == torch.bool
