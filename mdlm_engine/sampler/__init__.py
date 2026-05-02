"""Confidence-based samplers.

Public:
    Sampler                   — Protocol; the function signature
    register_sampler          — decorator
    get_sampler               — lookup by name

Built-in samplers (auto-registered on import of `confidence`):
    'argmax', 'maskgit_plus', 'entropy', 'margin', 'topk_margin'
"""
import importlib

from mdlm_engine.sampler.base import Sampler, get_sampler, register_sampler

# Side-effect import: each function in `confidence` is decorated with
# `@register_sampler`, which fires at import time. We don't need the
# module object itself, just the registrations — `import_module` makes
# this unambiguous to linters.
importlib.import_module("mdlm_engine.sampler.confidence")

__all__ = ["Sampler", "register_sampler", "get_sampler"]
