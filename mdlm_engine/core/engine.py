"""DiffusionEngine — public entrypoint.

Usage::

    from mdlm_engine import DiffusionEngine
    from mdlm_engine.adapters.dream import DreamAdapter

    engine = DiffusionEngine(
        model, adapter=DreamAdapter(model, tokenizer),
        cache="dkv", sampler="entropy", scheduler="slowfast",
    )
    out = engine.generate(prompt_ids, max_new_tokens=512, ...)
    print(tokenizer.decode(out.sequences[0]))

The engine is *the* model-agnostic layer. It selects cache / sampler /
scheduler by name, instantiates the right `DiffusionCache` for the model's
`cache_layout()`, and drives the per-block diffusion loop.

Phase 1 day 5: the engine plumbs adapter ↔ cache ↔ sampler ↔ scheduler
end-to-end. Real generation runs at day 8-9 acceptance gate (needs GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mdlm_engine.cache.base import DiffusionCache, NoOpCache
from mdlm_engine.cache.block import BlockCache
from mdlm_engine.cache.dkv import DKVCache
from mdlm_engine.core.loop import LoopConfig, generate_block
from mdlm_engine.core.state import GenerationState
from mdlm_engine.sampler.base import get_sampler
from mdlm_engine.scheduler.base import get_scheduler

if TYPE_CHECKING:
    from mdlm_engine.adapters.base import ModelAdapter


@dataclass
class GenerateOutput:
    """Return value of `DiffusionEngine.generate`."""

    sequences: torch.Tensor                    # [B, L_p + L_g] long
    num_forwards: int                           # total forward passes used
    trace: list = field(default_factory=list)   # populated if return_trace=True


class DiffusionEngine:
    """Model-agnostic engine for masked diffusion LM inference."""

    def __init__(
        self,
        model: "torch.nn.Module",
        *,
        adapter: "ModelAdapter",
        cache: str = "dkv",
        sampler: str = "entropy",
        scheduler: str = "slowfast",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.cache_kind = cache
        self.sampler_fn = get_sampler(sampler)
        self.scheduler_fn = get_scheduler(scheduler)

    # ------------------------------------------------------------------
    # Cache factory
    # ------------------------------------------------------------------

    def _build_cache(self, batch_size: int, max_length: int) -> DiffusionCache:
        """Allocate a `DiffusionCache` matching the adapter's K/V layout.

        Phase 1 supports three cache kinds:
            - 'none'  — no caching; every step recomputes everything.
            - 'block' — `BlockCache`; recompute the active block each step.
            - 'dkv'   — `DKVCache`; recompute only currently-masked positions.

        Phase 1 also assumes the adapter's K/V layout is HF_LEGACY_KV_BLD
        (raw (K, V) tuples with shape [B, n_kv, L, d_h]). Adapters that
        advertise a different layout will raise here so the abstraction
        leak is caught at construction, not at the first forward.
        """
        from mdlm_engine.adapters.base import CacheLayout

        layout = self.adapter.cache_layout()
        if layout != CacheLayout.HF_LEGACY_KV_BLD:
            raise NotImplementedError(
                f"Phase-1 caches only support CacheLayout.HF_LEGACY_KV_BLD; "
                f"adapter {type(self.adapter).__name__} reported {layout!r}. "
                f"Add layout-specific allocation to _build_cache before shipping "
                f"this adapter."
            )
        kwargs = dict(
            n_layers=self.adapter.n_layers,
            n_kv_heads=self.adapter.n_kv_heads,
            head_dim=self.adapter.head_dim,
            max_length=max_length,
            batch_size=batch_size,
            dtype=self._infer_dtype(),
            device=self._infer_device(),
        )
        if self.cache_kind == "none":
            return NoOpCache(**kwargs)
        if self.cache_kind == "block":
            return BlockCache(**kwargs)
        if self.cache_kind == "dkv":
            return DKVCache(**kwargs)
        raise ValueError(
            f"unknown cache='{self.cache_kind}'. Expected 'none', 'block', or 'dkv'."
        )

    def _infer_dtype(self) -> torch.dtype:
        for p in self.model.parameters():
            return p.dtype
        return torch.bfloat16

    def _infer_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int = 512,
        block_length: int = 32,
        steps_per_block: int = 32,
        temperature: float = 0.0,
        top_p: float | None = 0.95,
        top_k: int | None = None,
        confidence_threshold: float = 0.9,
        speculative_k: int = 1,
        speculative_threshold: float = 0.99,
        return_trace: bool = False,
    ) -> GenerateOutput:
        """Run masked-diffusion generation.

        Parameters
        ----------
        prompt_ids :
            ``[B, L_p]`` long tensor on the model's device.
        max_new_tokens :
            How many tokens to generate beyond the prompt. Must be a
            multiple of ``block_length``. Default 512 (v0.2.2 quality
            preset). Use 256 for speed if completion length permits;
            use 768 to match Dream-Coder's paper default.
        block_length :
            Tokens per diffusion block. Default 32 (matches Dream/DiffuCoder).
        steps_per_block :
            Denoising steps allocated per block. 32 = full quality, 16 = fast,
            4 = aggressive (only useful with a model trained for low-step).
        """
        if max_new_tokens % block_length != 0:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must be a multiple of "
                f"block_length ({block_length})."
            )

        device = prompt_ids.device
        B, L_p = prompt_ids.shape
        L_max = L_p + max_new_tokens

        # Pad x with mask tokens. For masked diffusion, attn_mask is 1 EVERYWHERE
        # — including at mask-token positions — because the model attends
        # bidirectionally over the entire sequence (prompt + masks + future).
        # Setting attn_mask=0 at masked positions makes the model ignore them
        # and produces gibberish. Verified convention from fast_dllm:
        #   modeling_dream.py:482-484  pads attention_mask with VALUE=1.0
        #   sft_dataset.py:147-149     `torch.ones(...)  # NOTE: we use 1 here`
        x = torch.full(
            (B, L_max), self.adapter.mask_token_id,
            dtype=torch.long, device=device,
        )
        x[:, :L_p] = prompt_ids
        attn_mask_1d = torch.ones(B, L_max, dtype=torch.long, device=device)

        cache = self._build_cache(batch_size=B, max_length=L_max)
        state = GenerationState(
            x=x,
            attn_mask_1d=attn_mask_1d,
            block_start=L_p,
            block_end=L_p + block_length,
            eos_seen=torch.zeros(B, dtype=torch.bool, device=device),
        )

        cfg = LoopConfig(
            block_length=block_length,
            steps_per_block=steps_per_block,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            speculative_k=speculative_k,
            speculative_threshold=speculative_threshold,
        )

        total_forwards = 0
        trace: list = []
        eos_set = set(self.adapter.eos_token_ids)

        # `inference_mode` disables autograd globally for everything inside.
        # Without it, PATH A's chain of cached forwards builds a graph that
        # holds references to every intermediate activation across all 32 ×
        # n_blocks steps — exhausts the 5090's 32 GB at iter step 1 of problem 0.
        # `model.eval()` alone does NOT disable autograd; it only flips dropout/
        # batchnorm. inference_mode is preferred over no_grad: it's strictly
        # stronger (zero version-counter overhead) and we never need autograd
        # during generation.
        n_blocks = max_new_tokens // block_length
        with torch.inference_mode():
            for block_idx in range(n_blocks):
                forwards = generate_block(
                    state=state, adapter=self.adapter, cache=cache,
                    sampler=self.sampler_fn, scheduler=self.scheduler_fn, cfg=cfg,
                )
                total_forwards += forwards
                if return_trace:
                    trace.append({
                        "block": block_idx,
                        "forwards": forwards,
                        "x_snapshot": state.x.clone().cpu(),
                    })

                # No attn_mask update needed: it's already all-1s by construction
                # (masked-diffusion convention; mask tokens are valid attendees).

                # Check EOS in the just-finalized block.
                block_slice = state.x[:, state.block_start:state.block_end]
                for tok_id in eos_set:
                    state.eos_seen |= (block_slice == tok_id).any(dim=-1)
                if bool(state.eos_seen.all()):
                    break

                state.block_start = state.block_end
                state.block_end = min(state.block_end + block_length, L_max)
                if state.block_end <= state.block_start:
                    break

        return GenerateOutput(
            sequences=state.x[:, :state.block_end].clone(),
            num_forwards=total_forwards,
            trace=trace,
        )
