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
      In-place K/V replacement at masked positions — the fast path. Requires
      the fast_dllm-patched ``modeling_dream.py``; the upstream HF Hub model
      is plain HF and does NOT include these kwargs.
    - PATH B: standard HF caching (``past_key_values`` + ``use_cache``) but no
      fast_dllm extensions. **Cannot speed up masked diffusion** — stock HF
      caching is append-only (``torch.cat([past, new], dim=-2)``), and masked
      diffusion needs in-place replace at masked positions. We therefore
      collapse PATH B → PATH C inside ``forward()`` with a one-time warning.
      The detection itself stays as 'B' so the caps record matches reality.
    - PATH C: no caching support. Match v0.1.0 behavior (``use_cache=False``).

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
        if self._caps.path != "A":
            import warnings
            if self._caps.path == "B":
                msg = (
                    "DreamAdapter: model.forward accepts past_key_values but "
                    "lacks fast_dllm's dual_cache/replace_position extensions. "
                    "Stock HF caching is append-only and cannot accelerate "
                    "masked diffusion (PATH B → PATH C fallback). Speed will "
                    "match v0.1.0 (~10 s/problem). For the v0.2.0 speedup, "
                    "load Dream from a fast_dllm-patched modeling_dream.py."
                )
            else:
                msg = (
                    "DreamAdapter: model.forward doesn't accept past_key_values. "
                    "Phase-2 cache wiring will fall back to use_cache=False, "
                    "matching v0.1.0 speed (~10 s/problem). Use a fast_dllm-patched "
                    "Dream-Coder modeling for the v0.2.0 speedup."
                )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

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
        # PATH B is detected accurately but collapsed to PATH C here — see
        # `_DreamCaps` docstring for why stock HF caching can't accelerate
        # masked diffusion (append-only torch.cat vs needed in-place replace).
        if use_cache and diffusion_cache is not None and self._caps.path == "A":
            wiring = "A"
        else:
            wiring = "C"

        if wiring == "A":
            # fast_dllm expects past_key_values per-layer as 3D `[B, L, H*D]` —
            # the K/V projections BEFORE the per-head view (modeling_dream.py:480-490).
            # Our cache stores HF-standard 4D `[B, H, L, D]`. Convert at the
            # boundary; aliasing through a permute is broken by reshape needing
            # contiguous memory, so we explicitly copy back into the cache after
            # the forward returns.
            legacy_4d = diffusion_cache.to_legacy_kv()
            pkv_3d: list[tuple[torch.Tensor, torch.Tensor]] = []
            for K_4d, V_4d in legacy_4d:
                B, H, L, D = K_4d.shape
                K_3d = K_4d.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)
                V_3d = V_4d.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)
                pkv_3d.append((K_3d, V_3d))
            kwargs["past_key_values"] = pkv_3d
            kwargs["use_cache"] = True
            kwargs["dual_cache"] = True
            # replace_position: True at positions the model should recompute
            # K/V for (= NOT yet committed). commit_state() is True at
            # frozen/committed positions; invert.
            kwargs["replace_position"] = ~diffusion_cache.commit_state()
        else:
            # PATH C (or PATH B collapsed to C): v0.1.0 behavior. No caching.
            kwargs["use_cache"] = False

        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logits = self.shift_logits(logits)

        if wiring == "A":
            # Write the model's updated 3D K/V back into the 4D cache. The
            # cache's _K/_V are storage-shaped `[B, H, L, D]`; we materialize
            # the inverse permute. This is one extra copy per layer per step
            # (~12 MB on Dream-7B) — bandwidth-bound, ~ms scale on Blackwell.
            updated_pkv = getattr(out, "past_key_values", None)
            if updated_pkv is not None:
                H = self.n_kv_heads
                D = self.head_dim
                for layer_idx, (K_3d, V_3d) in enumerate(updated_pkv):
                    B, L, _ = K_3d.shape
                    K_4d = K_3d.view(B, L, H, D).permute(0, 2, 1, 3).contiguous()
                    V_4d = V_3d.view(B, L, H, D).permute(0, 2, 1, 3).contiguous()
                    diffusion_cache._K[layer_idx].copy_(K_4d)
                    diffusion_cache._V[layer_idx].copy_(V_4d)
            return AdapterOutput(logits=logits, past_key_values=None)

        return AdapterOutput(logits=logits, past_key_values=None)

    # ------------------------------------------------------------------
    # Cache layout
    # ------------------------------------------------------------------

    def cache_layout(self) -> CacheLayout:
        return CacheLayout.HF_LEGACY_KV_BLD
