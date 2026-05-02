"""Shared pytest configuration / fixtures.

Conventions:
    - `@pytest.mark.gpu` skips when CUDA isn't available.
    - `@pytest.mark.slow` opts in to >30s tests.
    - `@pytest.mark.adapter_validation` marks the four-test contract that every
      registered adapter must pass.
"""
from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu when CUDA is unavailable."""
    skip_gpu = pytest.mark.skip(reason="requires CUDA (use @pytest.mark.gpu)")
    has_torch = importlib.util.find_spec("torch") is not None
    cuda_available = False
    if has_torch:
        import torch  # type: ignore

        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
