"""The four-test adapter validation contract.

Every `ModelAdapter` subclass shipped with `mdlm-engine` MUST pass these four
tests before being registered. The contract enforces that the abstraction
doesn't rot as adapters are added.

Phase 1 status: contract is wired up, GPU-bound tests are deferred (they need
the engine to be implementable). The contract becomes blocking on Phase 1
day 8 when the engine ships its first end-to-end generate path.

Per plan §"Mandatory adapter validation":

    1. Logit alignment — argmax(shift_logits(logits)[i]) predicts token i
       on a known prompt at temp 0.
    2. Round-trip equivalence — engine vs reference, max-abs logit diff < 1e-3.
    3. Cache equivalence — cache=none vs block vs dkv all match within tol.
    4. NaN-freedom — 100 generations at temp 0, zero NaN logits.
"""
from __future__ import annotations

import pytest


@pytest.mark.adapter_validation
def test_validation_helpers_importable():
    """The four-test runner must be importable without GPU."""
    from mdlm_engine.bench.equivalence import (
        DEFAULT_SAMPLE_PROMPTS,
        run_adapter_validation,
        write_validation_report,
    )

    assert len(DEFAULT_SAMPLE_PROMPTS) >= 5
    assert callable(run_adapter_validation)
    assert callable(write_validation_report)


@pytest.mark.adapter_validation
@pytest.mark.gpu
@pytest.mark.skip(reason="Phase 1 day 8: GPU-bound, needs engine to be implementable")
def test_dream_adapter_passes_contract():
    """Dream-Coder adapter must pass all four mandatory tests."""
    from transformers import AutoModel, AutoTokenizer

    from mdlm_engine.adapters.dream import DreamAdapter

    model = AutoModel.from_pretrained(
        "Dream-org/Dream-Coder-v0-Instruct-7B",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-Coder-v0-Instruct-7B",
        trust_remote_code=True,
    )
    adapter = DreamAdapter(model=model, tokenizer=tok)
    report = adapter.validate()

    assert report.all_passed, f"Dream adapter failed validation: {report}"


@pytest.mark.adapter_validation
@pytest.mark.gpu
@pytest.mark.skip(reason="Phase 1 day 8: GPU-bound, needs engine to be implementable")
def test_llada_adapter_passes_contract():
    """LLaDA adapter must pass all four mandatory tests.

    Critical for the model-agnostic claim: if LLaDA passes the same contract
    as Dream using the same engine code, the abstraction is honest.
    """
    from transformers import AutoModel, AutoTokenizer

    from mdlm_engine.adapters.llada import LLaDAAdapter

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
    )
    adapter = LLaDAAdapter(model=model, tokenizer=tok)
    report = adapter.validate()

    assert report.all_passed, f"LLaDA adapter failed validation: {report}"
