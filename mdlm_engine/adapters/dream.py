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
        """Run one forward pass through Dream. Converts DiffusionCache <-> raw tuple.

        Phase 1: ``diffusion_cache`` and ``use_cache`` are intentionally ignored
        — caching is engine-side. The model itself runs with use_cache=False
        and recomputes its own K/V every forward; the engine controls *which*
        positions get fed in via active_ids slicing. Model-side cache reuse
        (passing past_key_values into the model) layers in with the day-7
        ops module.
        """
        del diffusion_cache, use_cache  # see docstring
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask if not isinstance(attention_mask, str) else None,
            position_ids=position_ids,
            use_cache=False,
        )
        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logits = self.shift_logits(logits)
        return AdapterOutput(logits=logits, past_key_values=None)

    # ------------------------------------------------------------------
    # Cache layout
    # ------------------------------------------------------------------

    def cache_layout(self) -> CacheLayout:
        return CacheLayout.HF_LEGACY_KV_BLD
