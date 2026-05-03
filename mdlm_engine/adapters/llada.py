r"""LLaDAAdapter — bridge for `GSAI-ML/LLaDA-8B-Base` and family.

This adapter is the abstraction stress test: same engine code as Dream, but
LLaDA has DIFFERENT conventions across the board. If both adapters work
through the same `DiffusionEngine` without engine code changes, the
model-agnostic claim holds empirically.

LLaDA-vs-Dream conventions (verified via `scripts/day1_spike/01_llada_spike.json`):

| Convention             | Dream-Coder              | LLaDA-8B-Base           |
|------------------------|--------------------------|-------------------------|
| Mask token             | `<\|mask\|>` (151666)    | `<\|mdm_mask\|>` (126336) |
| Pad/EOS                | 151643/151645            | `<\|endoftext\|>` (126081) |
| `position_ids` accepted| yes                      | NO (not in forward)     |
| Logit shift            | shift right by 1         | identity (no shift)     |
| Attention mask shape   | 4D bool                  | 2D long (LLaDA broadcasts internally) |
| Extra forward kwargs   | none we use              | `attention_bias` (we pass None) |

So the adapter for LLaDA returns:
  - build_position_ids: None (forward doesn't accept position_ids anyway)
  - build_attention_mask: 2D long tensor (LLaDA's modeling expects [B, L])
  - shift_logits: identity

That's it. ~80 LOC vs Dream's ~150.
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


LLADA_MASK_TOKEN = "<|mdm_mask|>"
LLADA_MASK_TOKEN_ID = 126336
LLADA_PAD_EOS = "<|endoftext|>"
LLADA_PAD_EOS_ID = 126081


@register_adapter("llada")
class LLaDAAdapter(ModelAdapter):
    """Adapter for LLaDA-8B-Base / -Instruct."""

    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__(model=model, tokenizer=tokenizer)
        mask_id = tokenizer.convert_tokens_to_ids(LLADA_MASK_TOKEN)
        if mask_id is None or mask_id == tokenizer.unk_token_id:
            raise ValueError(
                f"LLaDAAdapter: tokenizer doesn't know '{LLADA_MASK_TOKEN}'. "
                f"Got mask_id={mask_id}, expected {LLADA_MASK_TOKEN_ID}."
            )
        if mask_id != LLADA_MASK_TOKEN_ID:
            import warnings
            warnings.warn(
                f"LLaDAAdapter: mask_token_id is {mask_id}, expected "
                f"{LLADA_MASK_TOKEN_ID}. Continuing but verify outputs.",
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
            raise ValueError(
                "LLaDAAdapter: tokenizer has neither pad_token_id nor eos_token_id"
            )
        return int(pad)

    @property
    def eos_token_ids(self) -> list[int]:
        ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        # LLaDA uses <|endoftext|> as both pad and eos.
        for tok in (LLADA_PAD_EOS, "<|eot_id|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id and tid not in ids:
                ids.append(int(tid))
        return ids

    # ------------------------------------------------------------------
    # Canonical input
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> torch.Tensor:
        """LLaDA uses Llama-3-style ChatML; defer to the tokenizer's template."""
        result = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        return result["input_ids"]

    # ------------------------------------------------------------------
    # Forward conventions — LLaDA differs from Dream HERE
    # ------------------------------------------------------------------

    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask_1d: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """LLaDA's forward signature lacks position_ids — return None.

        Verified in scripts/day1_spike/01_llada_spike.json:
            forward_accepts.position_ids = false
        """
        del input_ids, attention_mask_1d  # ABC-mandated args; LLaDA doesn't use them
        return None

    def build_attention_mask(
        self,
        attention_mask_1d: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | str:
        """LLaDA accepts a standard 2D attention mask `[B, L]`."""
        del seq_len  # LLaDA derives shape from attention_mask_1d directly
        if attention_mask_1d is None:
            return "bidirectional"
        return attention_mask_1d

    def shift_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Identity. Verified via the day-1 spike's logit_shift_test:
        decoded_no_shift='1' is the plausible continuation for
        'def fib(n): return ', so LLaDA does NOT need a shift."""
        return logits

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
        """One forward through LLaDA. We do NOT pass position_ids
        (LLaDA's forward doesn't accept it), and we pass attention_bias=None
        (LLaDA-specific; default behavior).

        **v0.2.0 cache deferral.** The day-1 spike confirmed LLaDA's forward
        accepts ``past_key_values`` + ``use_cache`` but NOT fast_dllm's
        ``dual_cache`` / ``replace_position`` extensions
        (`scripts/day1_spike/01_llada_spike.json:forward_accepts`). Standard
        HF caching is semantically awkward for masked diffusion — past_key_values
        is append-only (`torch.cat([past, new], dim=-2)`), but diffusion needs
        in-place replace at masked positions. Without the fast_dllm extensions
        we can't safely reuse the cache, so we keep ``use_cache=False`` and
        ignore the engine's request. LLaDA still runs end-to-end at v0.1.0
        speed; full caching support is a v0.2.1 follow-up that requires either
        patching LLaDA's modeling code with fast_dllm-style extensions or
        building a per-step write-back path through ``update_from_model_output``.
        """
        del position_ids, diffusion_cache, use_cache  # see docstring
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask if not isinstance(attention_mask, str) else None,
            use_cache=False,
        )
        out = self.model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logits = self.shift_logits(logits)  # identity for LLaDA
        return AdapterOutput(logits=logits, past_key_values=None)

    # ------------------------------------------------------------------
    # Cache layout + GQA handling
    # ------------------------------------------------------------------

    def cache_layout(self) -> CacheLayout:
        return CacheLayout.HF_LEGACY_KV_BLD

    @property
    def n_kv_heads(self) -> int:
        """LLaDA's config sets `num_key_value_heads = None` (not absent), so
        the ABC's getattr-with-default doesn't trigger. Override with an
        explicit None check.
        """
        cfg = self.model.config
        nkv = getattr(cfg, "num_key_value_heads", None)
        if nkv is None:
            return int(cfg.num_attention_heads)
        return int(nkv)
