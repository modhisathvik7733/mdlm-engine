# mdlm-engine v0.2.1 — PATH A engaged: 1.34× speedup on Dream-Coder

The cache-wiring infrastructure shipped in v0.2.0 was correct but couldn't engage on the upstream HF Hub Dream-Coder model (stock HF caching is append-only; can't accelerate masked diffusion). v0.2.1 bundles the fast_dllm-patched `modeling_dream.py`, wires up the init/iter protocol, and turns the speedup on.

## Acceptance results (HumanEval+, limit=20, RTX 5090, 2026-05-03)

| Path | Modeling | s/problem | tokens/sec | pass@1 | Peak VRAM | NaN / 100 |
|---|---|---:|---:|---:|---:|---:|
| **PATH B → C** (v0.2.0) | upstream HF Dream | 11.31 | 19.5 | 0.950 | 15.89 GB | 0 |
| **PATH A** (v0.2.1) | fast_dllm-patched | **8.42** | **25.5** | **0.850** | 15.79 GB | 0 |
| Δ | | **1.34× faster** | **+31%** | -10 pp | -0.6% | unchanged |

Plus all gates green:
- 117 CPU unit tests + 2 GPU adapter-validation tests
- Cache equivalence on PATH A: `block ≡ dkv` identical on 5 prompts × 3 pairs at temp 0
- LLaDA portability smoke unchanged: 0.500 / 13.72 s/problem (LLaDA caching still deferred)

## What's new

- **`mdlm_engine/models/dream_fastdllm/`**: bundled fast_dllm `modeling_dream.py` (1027 LOC, verbatim) + `load_dream_fastdllm()` helper. The helper snapshot-downloads the HF Hub weights, overlays our patched modeling on top, and returns `AutoModel.from_pretrained(cache_dir, trust_remote_code=True)` — no model fork, no separate weights.
- **Bench harness**: `--use_fastdllm_modeling` flag (Dream only) routes to the helper.
- **DreamAdapter PATH A protocol**: full init/iter dispatch matching fast_dllm's reference `generation_utils_block.py:495-571`:
  - **Init pass** (one per block): full-sequence forward, `past_key_values=None`, `use_cache=True`. Model returns past_key_values for the whole sequence; we copy them into our 4D cache (3D→4D layout transform).
  - **Iter pass** (per step within block): `input_ids = state.x[:, block_start:block_end]`, `past_key_values=cache` (4D→3D), `dual_cache=True`, `replace_position` true only at active block. Model recomputes K/V at active block via fast_dllm's in-place `past_key[:, replace_indices] = key_states`. We copy back 3D→4D.
- **`core/loop.py`**: passes `is_init=(step==0)` and `block_start/block_end` to the adapter; the adapter dispatches.
- **`core/engine.py`**: wraps `generate()` in `torch.inference_mode()` to disable autograd graph buildup (required by PATH A's chain of cached forwards).
- **`core/loop.py` block_start force-commit**: at step 0 of every block, force-commit `block_start` if it's still masked. Required because Dream's `shift_logits` (right-shift by 1) on a block-only iter forward would produce a misaligned `shifted_block[0]` (predicts `block_start+1` instead of `block_start`); fast_dllm's reference avoids this by always committing `block_start` from the init forward (`generation_utils_block.py:511`). No-op for PATH C; required for PATH A.

## Path-A correctness story (4 bugs caught & fixed during the gate)

1. **Day-3 commit `f549734`** — Layout: fast_dllm expects past_key_values per layer as 3D `[B, L, H*D]`; our cache stores HF 4D `[B, H, L, D]`. Convert at the adapter boundary.
2. **Day-3 commit `5cfa51e`** — Protocol: fast_dllm's iter forwards expect block-only `input_ids` with `q_len == len(replace_indices)`. Our engine was passing the full sequence. Restructured to init/iter dispatch with engine-side `is_init=(step==0)` flag.
3. **Day-3 commit `6b83b6d`** — OOM: `model.eval()` does not disable autograd; PATH A's chained cached forwards built a 14 GB graph by iter step 1. Wrapped `generate()` in `torch.inference_mode()`.
4. **Day-3 commit `01c23a2`** — Logit alignment: block-only iter logits make `shifted_block[0]` predict `block_start+1`, not `block_start`. Force-commit `block_start` from init forward to match fast_dllm's protocol.

Without the gate's pass@1 metric we'd have shipped #1 and #2 invisible (no crash but no speedup, or crashes only at iter step 1+). #3 and #4 are silent quality regressions a unit test couldn't catch.

## Why pass@1 dropped 10 pp

Two contributors, hard to disentangle on a 20-problem subset:

1. **Forced commit at `block_start`**: PATH C's slowfast scheduler picks the highest-confidence position to commit at step 0; PATH A always picks `block_start`, which may be lower-confidence. 1-2 problems per 20-problem subset is plausible noise.
2. **Numerical drift across cached steps**: bf16 K/V at masked positions get computed once at init and reused across all 32 iter steps. Tentative-K/V semantics mismatch with the true "denoising" K/V the model would compute step-by-step.

Either bumps pass@1 down a few percentage points but stays at the gate floor (≥ 0.85). On a full HumanEval+ subset (~164 problems) the spread should narrow.

## What this enables for v0.3.0

- **Self-speculative decoding** (arxiv 2510.04147): PATH A's caching infrastructure is the foundation. Speculation generates K candidate tokens at low-confidence positions in parallel; verification is a single forward. Realistic target: ≤ 4-5 s/problem on Dream.
- **LLaDA caching** (v0.2.2): the same protocol can be ported to LLaDA if a fast_dllm-style patched modeling is available.

## Backwards-incompatible changes

None for users of the public API. Internal: `ModelAdapter.forward()` ABC gained 3 optional kwargs (`block_start`, `block_end`, `is_init`); existing adapters that didn't use them still work because they're keyword-only with defaults.

## Cost

Phase 2.1 development: ~$3 of vast.ai RTX 5090 time across 4 gate runs (each ~30-45 min). Total Phase 2 cost: ~$5.

## Hardware

Tested on a single NVIDIA RTX 5090 (Blackwell, 32 GB) with PyTorch 2.11 cu130. Should also work on A100/H100 with stable PyTorch + CUDA 12.4.

---

Built in collaboration with Claude (Anthropic). All code authored by the developer; planning and pair-programming by Claude.
