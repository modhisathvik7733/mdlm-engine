"""Step / commit schedulers.

Public:
    Scheduler                 — Protocol; the function signature
    register_scheduler        — decorator
    get_scheduler             — lookup by name

Built-in (auto-registered):
    'uniform'        — equal tokens-per-step (Dream-Coder default style)
    'confidence'     — threshold-based (fast_dllm dual_cache style)
    'slowfast'       — two-phase explore-then-exploit (arxiv 2506.10848)
"""
import importlib

from mdlm_engine.scheduler.base import Scheduler, get_scheduler, register_scheduler

# Side-effect imports register schedulers via @register_scheduler.
for _mod in ("uniform", "confidence", "slowfast"):
    importlib.import_module(f"mdlm_engine.scheduler.{_mod}")
del _mod

__all__ = ["Scheduler", "register_scheduler", "get_scheduler"]
