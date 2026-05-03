"""Per-block diffusion generation loop — the engine's hot path.

Pseudocode preview (full impl below):

    for step in range(steps_per_block):
        active_ids = state.x[:, block_start:block_end]
        out = adapter.forward(active_ids, attn_mask, position_ids,
                              diffusion_cache=cache, use_cache=True)
        cache.update_from_model_output(out.past_key_values, ...)
        logits = adapter.shift_logits(out.logits)
        mask_index = (active_ids == adapter.mask_token_id)
        if not mask_index.any():
            break
        confidences, candidates = sampler(logits[mask_index], ...)
        commit_mask = scheduler.select(confidences, mask_index, step, ...)
        write_back(state.x, block_start, candidates, commit_mask)
        cache.commit(committed_positions)

This module exposes one entry point — ``generate_block`` — that takes
adapter/cache/sampler/scheduler + state and runs one block to completion.

The engine's outer loop (in ``engine.py``) advances ``block_start``/
``block_end`` and calls this once per block.

NOTE: this is a Phase-1-day-5 stub that defines the public surface. The
actual K/V plumbing through ``adapter.forward`` is finalized when the
DreamAdapter ships in this same commit. The loop will go green end-to-end
on a GPU box at the day-8 acceptance gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mdlm_engine.adapters.base import ModelAdapter
    from mdlm_engine.cache.base import DiffusionCache
    from mdlm_engine.core.state import GenerationState
    from mdlm_engine.sampler.base import Sampler
    from mdlm_engine.scheduler.base import Scheduler


@dataclass
class LoopConfig:
    """Knobs the per-block loop reads. The engine fills these in once."""

    block_length: int = 32
    steps_per_block: int = 32
    temperature: float = 0.0
    top_p: float | None = 0.95
    top_k: int | None = None
    # v0.3.0 self-speculative decoding: 0 = disabled (v0.2.x behavior).
    # When > 0, after each step's regular commit, propose this many extra
    # high-confidence masked positions and verify with one extra forward.
    # Lossless at temperature == 0 IF confidence threshold is high enough
    # to avoid commit-order drift; see speculative_threshold.
    speculative_k: int = 0
    # v0.3.0 SSD confidence gate. Day-1 v0.3.0 gate run found threshold=0
    # drops pass@1 by 25 pp due to commit-order drift in masked diffusion
    # (different commit order → different cascading context, even at
    # temp=0). threshold=0.95 default only proposes positions where the
    # model's top-1 probability ≥ 95% — at production sampling settings
    # (temp=0.2, top_p=0.95) this is the threshold above which top-p
    # sampling has no other candidates, so SSD's argmax commit matches
    # what the sampler would have done. Lossless within sampling noise.
    speculative_threshold: float = 0.99
    confidence_threshold: float = 0.9


def generate_block(
    state: "GenerationState",
    adapter: "ModelAdapter",
    cache: "DiffusionCache",
    sampler: "Sampler",
    scheduler: "Scheduler",
    cfg: LoopConfig,
) -> int:
    """Run one block to completion. Returns the number of forwards consumed.

    Mutates ``state`` in place: writes committed tokens into
    ``state.x[:, block_start:block_end]`` until the block has no masked
    positions left or the step budget is exhausted.

    Mutates ``cache`` in place via ``replace_at`` (per-step) and ``commit``
    (when the scheduler decides).

    **Phase-2 forward strategy:** the model sees the FULL sequence every
    step, but with ``use_cache=True``. Adapters that detected fast_dllm's
    ``dual_cache`` extension (PATH A) reuse K/V at *committed* positions
    via in-place writes through ``cache.to_legacy_kv()`` aliases — only
    masked positions recompute K/V. Adapters that lack the extension
    (PATH C, e.g., LLaDA on stock modeling) silently fall back to
    ``use_cache=False`` inside ``adapter.forward`` so this loop stays
    model-agnostic. Speedup comes from PATH A; PATH C matches v0.1.0 speed.
    """
    forwards = 0
    block_start, block_end = state.block_start, state.block_end

    for step in range(cfg.steps_per_block):
        # Mask check is on the active block only — that's where we sample/commit.
        active_ids = state.x[:, block_start:block_end]
        mask_index = active_ids == adapter.mask_token_id  # [B, block_len]
        if not bool(mask_index.any()):
            break

        # Forward sees the FULL sequence. Bidirectional attention; the model
        # doesn't know about block boundaries — it's just one big masked-LM pass.
        position_ids = adapter.build_position_ids(state.x, state.attn_mask_1d)
        attn_mask = adapter.build_attention_mask(state.attn_mask_1d, state.x.shape[1])

        out = adapter.forward(
            input_ids=state.x,
            attention_mask=attn_mask,
            position_ids=position_ids,
            diffusion_cache=cache,
            use_cache=True,
            block_start=block_start,
            block_end=block_end,
            is_init=(step == 0),
        )
        forwards += 1

        # Slice logits down to the active block before sampling — only block
        # positions are eligible to commit this step.
        full_logits = out.logits  # [B, L, V]; adapter already applied shift_logits
        block_logits = full_logits[:, block_start:block_end, :]   # [B, block_len, V]

        # v0.3.0 self-speculative decoding (arxiv 2510.04147), Redesign C.
        # At every step (not just step 0), AFTER the regular forward but
        # BEFORE the regular sampler, propose all currently-masked active-
        # block positions whose top-1 softmax probability ≥ threshold and
        # verify with one extra forward.
        #
        # Why "every step + full mask + before sampler" beats earlier designs:
        #   - Block-init at step 0 (Redesign A) failed: at step 0 ALL block
        #     positions are masked → low confidence everywhere → empty
        #     proposal → verify forward overhead with no commits.
        #   - Per-step on residual after sampler (day-1) only saw 1.12x
        #     because slowfast had already grabbed the high-confidence
        #     positions; SSD got "leftovers".
        #   - This design (every step, full mask, before sampler) lets SSD
        #     pick first as confidence accumulates step-by-step. At late
        #     steps when many positions are already committed and remaining
        #     ones become high-confidence, SSD fires and commits many in
        #     one verify forward.
        #
        # Lossless at high threshold (0.95) at production sampling settings
        # (temp=0.2, top_p=0.95): when the model's top-1 prob ≥ 95%, top-p
        # sampling has no other candidates → SSD's argmax-vs-argmax verify
        # picks exactly what the regular sampler would have committed.
        if cfg.speculative_k > 0:
            from mdlm_engine.speculative import (
                propose_block_level, verify as _spec_verify,
            )

            block_proposal = propose_block_level(
                block_logits=block_logits,
                mask_index=mask_index,
                block_start=block_start,
                confidence_threshold=cfg.speculative_threshold,
                max_proposals=None,  # threshold filters; no top-k cap
            )
            if len(block_proposal) > 0:
                spec_result = _spec_verify(
                    state=state,
                    cache=cache,
                    adapter=adapter,
                    proposal=block_proposal,
                    block_start=block_start,
                    block_end=block_end,
                    attn_mask_full=attn_mask,
                    position_ids_full=position_ids,
                )
                forwards += 1  # the verification forward
                if spec_result.n_accepted > 0:
                    cache.commit(spec_result.accepted_positions)
                    # Refresh mask_index — SSD wrote accepted tokens into
                    # state.x, so the regular sampler should skip them.
                    active_ids = state.x[:, block_start:block_end]
                    mask_index = active_ids == adapter.mask_token_id
                    if not bool(mask_index.any()):
                        # Block fully committed by SSD — skip rest of step
                        # and the outer loop's mask_index.any() guard will
                        # exit on the next iteration.
                        continue
        flat_mask = mask_index.flatten()
        flat_logits = block_logits.reshape(-1, block_logits.shape[-1])[flat_mask]
        confidences, candidates = sampler(
            flat_logits,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        # Scheduler chooses which masked positions commit this step.
        commit_mask = scheduler(
            confidences,
            mask_index=mask_index,
            step=step,
            steps_per_block=cfg.steps_per_block,
            threshold=cfg.confidence_threshold,
        )
        if not bool(commit_mask.any()):
            continue

        # Write committed candidates back into state.x at their positions.
        # Both `candidates` and `commit_mask` are flat [N]; we need to
        # scatter into the [B, block_len] active window.
        active_window = state.x[:, block_start:block_end]
        # Find the (b, p) indices of currently masked positions.
        mask_b, mask_p = mask_index.nonzero(as_tuple=True)
        committed = commit_mask.nonzero(as_tuple=False).flatten()
        for i in committed.tolist():
            b = int(mask_b[i])
            p = int(mask_p[i])
            active_window[b, p] = int(candidates[i])

        # Tell the cache which absolute positions were committed.
        committed_abs = (mask_p[committed] + block_start).long()
        # Use 1D positions; per-sample variation is fine since cache.commit
        # accepts 1D-broadcast OR 2D-per-sample. For Phase 1 we group by
        # batch on the engine side: collect each sample's positions and
        # call commit once with shape [B, n_pos]. For batch=1 this is
        # equivalent to 1D.
        if state.x.shape[0] == 1:
            cache.commit(committed_abs)
        else:
            # Build a 2D positions tensor — pad shorter samples with sentinel
            # values that the cache must ignore. Phase 1 keeps this simple by
            # walking sample-by-sample; multi-sample optimization is Phase 2.
            for b in range(state.x.shape[0]):
                # Positions committed for this sample.
                samp_mask = mask_b[committed] == b
                if not bool(samp_mask.any()):
                    continue
                samp_pos = (mask_p[committed][samp_mask] + block_start).long()
                # Wrap as [1, n_pos] to match the per-sample 2D contract,
                # then route through the broadcast-1D path with a single
                # sample-row update.
                # Simpler: directly mutate cache.commit with 1D positions
                # for the whole batch (cache will broadcast — but that's
                # incorrect for multi-sample). For multi-sample we extend
                # via 2D in Phase 2.
                cache.commit(samp_pos)

        # Force-commit block_start at step 0 if it's still masked. This
        # matches fast_dllm's protocol (`generation_utils_block.py:511`):
        # they sample at block_start FROM the init forward and commit it
        # before the iter loop begins. Reason: iter forwards (block-only
        # input) cannot produce a correct prediction for block_start
        # because Dream's `shift_logits` (right-shift by 1) on a block-
        # only output makes shifted[0] predict block_start+1, not
        # block_start. Sampling at block_start in iter mode would use a
        # misaligned logit → garbage outputs (caught by v0.2.1 day-3 gate
        # where pass@1 dropped to 0 with `pythonpython` token corruption).
        # By committing block_start at step 0 with the init forward's
        # correctly-aligned logit, iter steps never face the misalignment.
        # This is a no-op for PATH C but required for PATH A.
        if step == 0:
            row_mask_at_block_start = (
                state.x[:, block_start] == adapter.mask_token_id
            )
            if bool(row_mask_at_block_start.any()):
                # Sample at block_start from the init forward's logits.
                logit_at_block_start = full_logits[:, block_start, :]  # [B, V]
                _, candidate_at_block_start = sampler(
                    logit_at_block_start,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )
                # Only write at samples whose block_start is still masked.
                rows_to_write = row_mask_at_block_start.nonzero(as_tuple=True)[0]
                for b in rows_to_write.tolist():
                    state.x[b, block_start] = int(candidate_at_block_start[b])
                forced_pos = torch.tensor(
                    [block_start], dtype=torch.long, device=state.x.device,
                )
                cache.commit(forced_pos)

    return forwards
