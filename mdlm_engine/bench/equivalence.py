"""Adapter validation contract + logit-equivalence helpers.

The four-test adapter validation contract (per plan §"Mandatory adapter validation"):

    1. Logit alignment — argmax(shift_logits(logits)[i]) matches the i-th
       committed token, on a known prompt at temperature 0.
    2. Round-trip equivalence — engine vs reference `model.diffusion_generate(...)`,
       max-abs logit diff < 1e-3 over 5 prompts.
    3. Cache equivalence — `cache="none"` vs `cache="block"` vs `cache="dkv"`
       all produce the same logits within 1e-3 max-abs diff.
    4. NaN-freedom — 100 generations at temp 0, zero NaN logits.

This module ships the runtime helpers; `tests/test_adapter_validation.py`
hooks them into pytest.

These tests are mandatory: an adapter that fails any of them MUST NOT be
registered. The validation report is what catches "logit-shift bug looks
fluent but drops pass@1 by 5pp" before it reaches a real benchmark.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdlm_engine.adapters.base import ModelAdapter, ValidationReport


DEFAULT_SAMPLE_PROMPTS: list[str] = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write quicksort(arr) in Python using the Lomuto partition scheme.",
    "Implement an LRUCache class with O(1) get and put.",
    "Reverse a linked list iteratively in Python.",
    "Find the maximum subarray sum (Kadane's algorithm).",
]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_adapter_validation(
    adapter: "ModelAdapter",
    sample_prompts: list[str] | None = None,
    *,
    n_nan_check_runs: int = 100,
    logit_tol: float = 1e-3,
) -> "ValidationReport":
    """Run the four-test contract.

    Returns a `ValidationReport` with per-test pass/fail. Caller decides
    whether to proceed (pytest fails the test, runtime callers can branch).

    Imports are deferred so importing this module doesn't drag in torch.
    """
    from mdlm_engine.adapters.base import ValidationReport

    prompts = sample_prompts or DEFAULT_SAMPLE_PROMPTS
    notes: list[str] = []
    measurements: dict[str, float] = {}

    # Phase 1 stub: tests are wired up but not yet implemented (engine doesn't
    # exist yet). Each test currently records "deferred" so the contract is
    # visible in the report. Implementations land in subsequent commits.
    test1 = _logit_alignment_test(adapter, prompts, notes, measurements, logit_tol)
    test2 = _roundtrip_equivalence_test(adapter, prompts, notes, measurements, logit_tol)
    test3 = _cache_equivalence_test(adapter, prompts, notes, measurements, logit_tol)
    test4 = _nan_freedom_test(adapter, prompts, notes, measurements, n_nan_check_runs)

    return ValidationReport(
        adapter_name=adapter.__class__.__name__,
        model_type=adapter.model_type,
        logit_alignment_passed=test1,
        roundtrip_equivalence_passed=test2,
        cache_equivalence_passed=test3,
        nan_freedom_passed=test4,
        notes=notes,
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Individual tests (Phase 1 stubs — implementations land with engine)
# ---------------------------------------------------------------------------


def _logit_alignment_test(
    adapter: "ModelAdapter",
    prompts: list[str],
    notes: list[str],
    measurements: dict[str, float],
    tol: float,
) -> bool:
    """Test 1: argmax(shift_logits(logits)[i]) predicts token at position i.

    Known-good prompt: "def fib(n): return ". After shift, the logit at the
    last prompt position should argmax to a plausible next token (e.g. "n",
    "fib", "1"). If `shift_logits` is wrong (identity when shift is needed,
    or vice versa), this argmax will be off by one position and decode
    nonsense.

    Phase 1 stub: deferred until adapter.forward() is implementable.
    """
    notes.append("logit_alignment_test: NOT YET IMPLEMENTED (Phase 1 day 5)")
    return False  # deferred


def _roundtrip_equivalence_test(
    adapter: "ModelAdapter",
    prompts: list[str],
    notes: list[str],
    measurements: dict[str, float],
    tol: float,
) -> bool:
    """Test 2: engine logits match `model.diffusion_generate(...)` reference.

    Generate the same prompt with our engine (cache=none, scheduler=uniform,
    sampler=argmax, temperature=0) and with the model's native
    `diffusion_generate(...)`. Compare logit traces token-by-token.
    Pass if `max_abs_diff < tol` on all 5 prompts.

    Phase 1 stub: deferred until DiffusionEngine is implementable.
    """
    notes.append("roundtrip_equivalence_test: NOT YET IMPLEMENTED (Phase 1 day 8)")
    return False


def _cache_equivalence_test(
    adapter: "ModelAdapter",
    prompts: list[str],
    notes: list[str],
    measurements: dict[str, float],
    tol: float,
) -> bool:
    """Test 3: cache=none vs cache=block vs cache=dkv all produce same logits.

    Run the same engine config three times, varying only the cache impl.
    Compare logit traces pairwise. Pass if all three are within `tol`.

    Phase 1 stub: deferred until BlockCache + DKVCache are implementable.
    """
    notes.append("cache_equivalence_test: NOT YET IMPLEMENTED (Phase 1 day 8)")
    return False


def _nan_freedom_test(
    adapter: "ModelAdapter",
    prompts: list[str],
    notes: list[str],
    measurements: dict[str, float],
    n_runs: int,
) -> bool:
    """Test 4: 100 generations at temp 0, zero NaN in any logit.

    Catches the bnb-int8-style class of bug where the model itself or the
    cache reuse produces NaN logits at low temperature.

    Phase 1 stub: deferred until the engine can generate.
    """
    notes.append(f"nan_freedom_test (n={n_runs}): NOT YET IMPLEMENTED (Phase 1 day 8)")
    return False


# ---------------------------------------------------------------------------
# Convenience: dump ValidationReport to the adapter's directory
# ---------------------------------------------------------------------------


def write_validation_report(
    report: "ValidationReport",
    out_dir: str,
) -> str:
    """Persist a ValidationReport as JSON for reproducibility.

    Path: `{out_dir}/{model_type}_validation.json`. Returns the path written.
    """
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{report.model_type}_validation.json")
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    return path
