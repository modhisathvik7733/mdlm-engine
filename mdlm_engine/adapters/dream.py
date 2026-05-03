"""DreamAdapter — bridge for Dream-Coder-v0-Instruct-7B and family.

Conventions encoded here (verified from Dream-Coder/instruct/src/inference/
fast_dllm/modeling_dream.py and the day-1 spike):

- ``mask_token_id``: 151666 (`<|mask|>`).
- ``shift_logits``: shift-right by one — ``cat([logits[:, :1], logits[:, :-1]], dim=1)``.
  Dream's training loss has logits at position i predicting token i+1.
- ``build_position_ids``: ``cumsum(attention_mask) - 1`` (integer, clamped at 0).
- ``build_attention_mask``: 4D bool ``[B, 1, L, L]`` constructed as the outer
  AND of the 1D mask with itself (bidirectional).
- ``forward``: passes ``DiffusionCache`` through as a raw (K, V) tuple via
  the adapter's `_to_legacy_kv` helper, since Dream's modeling code expects
  the legacy tuple-of-tuples layout.

Target ~150 LOC — measured at end of file.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mdlm_engine.adapters.base import (
    AdapterOutput,
    CacheLayout,
    ModelAdapter,
    register_adapter,
)

if TYPE_CHECKING:
    from mdlm_engine.cache.base import DiffusionCache


# ---------------------------------------------------------------------------
# Dream-Coder canonical constants (verified)
# ---------------------------------------------------------------------------

DREAM_MASK_TOKEN = "<|mask|>"
DREAM_MASK_TOKEN_ID = 151666


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DreamCaps:
    """Which optional kwargs does the loaded Dream model's `forward` accept?

    Drives the cache-wiring path inside ``DreamAdapter.forward``:

    - PATH A: full fast_dllm extensions (``dual_cache`` + ``replace_position``).
      Use the fastest path with in-place K/V replacement at masked positions.
    - PATH B: standard HF caching (``past_key_values`` + ``use_cache``) but no
      fast_dllm extensions. Fall back to alias-based reads only; speedup
      is smaller but still meaningful.
    - PATH C: no caching at all. Match v0.1.0 behavior (``use_cache=False``).

    Detection is one-time at adapter construction. ``inspect.signature``
    is cheap; we keep the result cached on ``self._caps``.

    See ``scripts/day1_phase2/verify_dual_cache.py`` for the spike that
    surfaces these caps on the actual HF Hub Dream-Coder model.
    """

    accepts_past_key_values: bool
    accepts_use_cache: bool
    accepts_dual_cache: bool
    accepts_replace_position: bool

    @property
    def path(self) -> str:
        """Return ``'A'``, ``'B'``, or ``'C'`` per the docstring above."""
        if self.accepts_dual_cache and self.accepts_replace_position and self.accepts_past_key_values:
            return "A"
        if self.accepts_past_key_values and self.accepts_use_cache:
            return "B"
        return "C"


def _inspect_dream_caps(model: object) -> _DreamCaps:
    """Inspect ``model.forward`` and return a ``_DreamCaps`` record.

    Pure read — never calls the model. Safe to run on CPU before weights load.
    """
    try:
        sig = inspect.signature(model.forward)  # type: ignore[attr-defined]
        params = sig.parameters
    except (AttributeError, ValueError, TypeError):
        # Some compiled / quantized models don't expose a clean signature.
        # Treat as PATH C (no cache support) — engine will fall back.
        params = {}
    return _DreamCaps(
        accepts_past_key_values="past_key_values" in params,
        accepts_use_cache="use_cache" in params,
        accepts_dual_cache="dual_cache" in params,
        accepts_replace_position="replace_position" in params,
    )


@register_adapter("dream")
class DreamAdapter(ModelAdapter):
    """Adapter for Dream-Coder-v0-Instruct-7B and architectural siblings."""

    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__(model=model, tokenizer=tokenizer)
        # Resolve mask token from the tokenizer; fail loud if it's wrong.
        mask_id = tokenizer.convert_tokens_to_ids(DREAM_MASK_TOKEN)
        if mask_id is None or mask_id == tokenizer.unk_token_id:
            raise ValueError(
                f"DreamAdapter: tokenizer doesn't know token '{DREAM_MASK_TOKEN}'. "
                f"Got mask_id={mask_id}, expected {DREAM_MASK_TOKEN_ID}."
            )
        if mask_id != DREAM_MASK_TOKEN_ID:
            # Soft warning: some Dream variants may renumber. Engine will
            # work but the user should know.
            import warnings
            warnings.warn(
                f"DreamAdapter: mask_token_id is {mask_id}, expected "
                f"{DREAM_MASK_TOKEN_ID}. Continuing but verify outputs.",
                RuntimeWarning, stacklevel=2,
            )
        self._mask_id = int(mask_id)

        # Detect which cache-wiring path this model supports. Dream's modeling
        # code may or may not include fast_dllm's `dual_cache` and
        # `replace_position` extensions — see scripts/day1_phase2/verify_dual_cache.py
        # for the spike that determines this on a real HF Hub model.
        self._caps = _inspect_dream_caps(model)
        if not self._caps.accepts_past_key_values:
            import warnings
            warnings.warn(
                "DreamAdapter: model.forward doesn't accept past_key_values. "
                "Phase-2 cache wiring will fall back to use_cache=False, "
                "matching v0.1.0 speed (~10 s/problem). Use a fast_dllm-patched "
                "Dream-Coder modeling for the v0.2.0 speedup.",
                RuntimeWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Vocab properties
    # ------------------------------------------------------------------

    @property
    def mask_token_id(self) -> int:
        return self._mask_id

    @property
    def pad_token_id(self) -> int:
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("DreamAdapter: tokenizer has neither pad_token_id nor eos_token_id")
        return int(pad)

    @property
    def eos_token_ids(self) -> list[int]:
        ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        # Dream's chat template terminator (im_end). Dream uses Qwen-style ChatML.
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id and tid not in ids:
                ids.append(int(tid))
        return ids

    # ------------------------------------------------------------------
    # Canonical input
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> torch.Tensor:
        """Render messages with Dream/Qwen ChatML template; return [B, L] ids."""
        result = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        return result["input_ids"]

    # ------------------------------------------------------------------
    # Forward conventions
    # ------------------------------------------------------------------

    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask_1d: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """``cumsum(attn_mask) - 1`` clamped at 0 — Dream's convention."""
        del input_ids  # ABC-mandated; Dream's positions derive from attn_mask only
        if attention_mask_1d is None:
            return None
        positions = torch.cumsum(attention_mask_1d, dim=-1) - 1
        return positions.clamp(min=0)

    def build_attention_mask(
        self,
        attention_mask_1d: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | str:
        """4D bool ``[B, 1, L, L]`` bidirectional, derived from 1D mask."""
        del seq_len  # Dream derives the 4D shape from the 1D mask itself
        if attention_mask_1d is None:
            return "bidirectional"
        m = attention_mask_1d.bool()
        # Outer AND: token i can attend to token j iff both are real.
        attn4d = torch.logical_and(
            m.unsqueeze(1).unsqueeze(-2),  # [B, 1, 1, L]
            m.unsqueeze(1).unsqueeze(-1),  # [B, 1, L, 1]
        )
        return attn4d  # [B, 1, L, L] bool

    def shift_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Right-shift by 1: logits[i] should predict token at position i."""
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    # ------------------------------------------------------------------
    # The forward call
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        position_ids: torch.Tensor | None,
        diffusion_cache: "DiffusionCache | None",
        use_cache: bool,
    ) -> AdapterOutput:
        """Run one forward pass through Dream.

        Cache-wiring path is chosen at construction time via ``self._caps``:

        - **PATH A** (full fast_dllm support): pass ``past_key_values``,
          ``dual_cache=True``, ``replace_position`` derived from
          ``~commit_state``. Model writes new K/V in-place at masked positions
          via fast_dllm's ``past_key[:, replace_indices] = key_states`` pattern
          (`modeling_dream.py:484-490`). The cache picks up the writes
          automatically because ``to_legacy_kv()`` returns aliases of the
          internal ``_K``/``_V`` tensors.
        - **PATH B** (standard HF caching only): pass ``past_key_values`` +
          ``use_cache=True`` but skip the fast_dllm extensions. Speedup smaller.
        - **PATH C** (no caching support): match v0.1.0 behavior. Pass
          ``use_cache=False``, same forward as before.
        """
        # Build base kwargs common to all paths.
        kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask if not isinstance(attention_mask, str) else None,
            position_ids=position_ids,
        )

        # Decide path: caps + caller's `use_cache` request both must permit caching.
        wiring = self._caps.path if (use_cache and diffusion_cache is not None) else "C"

        if wiring == "A":
            kwargs["past_key_values"] = diffusion_cache.to_legacy_kv()  # aliases
            kwargs["use_cache"] = True
            kwargs["dual_cache"] = True
            # replace_position: True at positions the model should recompute
            # K/V for (= NOT yet committed). commit_state() is True at
            # frozen/committed positions; invert.
            kwargs["replace_position"] = ~diffusion_cache.commit_state()
        elif wiring == "B":
            # Standard HF caching: pass past_key_values + use_cache=True, no
            # in-place extension. The model returns concat'd past_key_values;
            # we write those back via update_from_model_output below.
            kwargs["past_key_values"] = diffusion_cache.to_legacy_kv()
            kwargs["use_cache"] = True
        else:
            # PATH C: v0.1.0 behavior. No caching.
            kwargs["use_cache"] = False

        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logits = self.shift_logits(logits)
        returned_pkv = getattr(out, "past_key_values", None) if wiring != "C" else None
        return AdapterOutput(logits=logits, past_key_values=returned_pkv)

    # ------------------------------------------------------------------
    # Cache layout
    # ------------------------------------------------------------------

    def cache_layout(self) -> CacheLayout:
        return CacheLayout.HF_LEGACY_KV_BLD
