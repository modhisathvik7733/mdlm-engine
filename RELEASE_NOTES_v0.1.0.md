# mdlm-engine v0.1.0 — Phase 1 release

A model-agnostic Python inference engine for masked diffusion language models. Same engine code runs Dream-Coder and LLaDA through ~190-LOC adapters per model.

## Acceptance results (HumanEval+, limit=20, RTX 5090 + PyTorch nightly cu128)

| Model | pass@1 single-shot | s/problem | tokens/sec | Peak VRAM | Adapter LOC |
|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-Coder-v0-Instruct-7B` | **0.900** (18/20) | 9.9 | 22 | 22 GB | 185 |
| `GSAI-ML/LLaDA-8B-Base` | **0.500** (10/20) | 12.6 | 20 | 23 GB | 197 |

Both substantially exceed Phase-1 thresholds (Dream ≥0.55, LLaDA ≥0.40). Same `mdlm_engine.bench.harness` invocation runs both; only the `--adapter` flag differs.

Plus:
- **92 CPU unit tests** passing.
- **100-generation NaN-freedom** check passes (zero NaN logits at temp 0).
- **Adapter LOC** both ≤ 200 (target was ≤ 200).

## What's in the box

```
mdlm_engine/
├── adapters/             ModelAdapter ABC + Dream + LLaDA
├── cache/                DiffusionCache ABC + BlockCache + DKVCache + NoOpCache
├── core/                 DiffusionEngine + per-block loop + state
├── sampler/              5 samplers: argmax, maskgit_plus, entropy, margin, topk_margin
├── scheduler/            3 schedulers: uniform, confidence, slowfast (arxiv 2506.10848)
├── ops/                  torch.compile + FlexAttention helpers
└── bench/                acceptance-gate runner + adapter-validation contract
```

Adding a new diffusion LM = subclass `ModelAdapter`, implement six methods, register, pass the four-test validation contract. ~80-200 LOC of model-specific code; engine code unchanged.

## Phase-1 design decisions (from the v2 plan)

- **Integrate at HF's public `forward()` and `Cache` API** — no fork of modeling code (`fast_dllm` does, which is why it's not portable).
- **Internal `DiffusionCache` ABC**, not a subclass of `transformers.cache_utils.Cache` (HF Cache is append-only; dKV needs `replace_at(positions, K, V)`).
- **Single-shot pass@1 is the headline metric**, with best-of-N reported separately as the oracle.
- **Hard adapter validation contract** (logit alignment / round-trip / cache equivalence / NaN freedom) — refusal to register a leaky adapter.
- **MXFP8 quantization deferred** to Phase 2: torchao's `_C_mxfp8.so` failed to load on PyTorch nightly cu128 + Blackwell as of the Day-1 spike (2026-05-02).

## Known limitation that drives v0.2.0

v0.1.0 keeps the cache **engine-side**: `DiffusionCache` tracks commit state, but the adapter's `forward()` calls the model with `use_cache=False`. So K/V is recomputed every diffusion step. That's why we land at `fast_dllm`-without-`dual_cache` speed (~10-12 s/problem) instead of `dual_cache` speed (~8 s).

**Phase 2 wires `DiffusionCache` into the model's `past_key_values` arg** — should hit ≈4 s/problem on Dream alone, before self-speculative decoding stacks on top for another ~1.5-2× win.

## Bugs fixed during the acceptance gate (architecture-level — caught by Day 8-9 GPU run, not by 92 CPU unit tests)

1. **`eed4aa1`**: engine was passing block-sized `input_ids` (32 tokens) but full-sequence attention mask (~400 tokens). Dream's attention layer can't reconcile that. Fix: pass full `state.x` as `input_ids`; slice logits down to the active block after the forward.
2. **`e502c39`**: engine was initializing `attention_mask_1d` with 0s on mask-token positions. Masked-diffusion attends bidirectionally over the entire sequence including mask tokens — `attention_mask=0` there told the model to ignore them and produced gibberish completions. Fix: initialize all-1s, matching `fast_dllm modeling_dream.py:482-484` and `sft_dataset.py:147-149`.
3. **`f688d33`**: `_check_completion` used a stale `evalplus.evaluate.check_correctness` signature with `expected_output=None` (not a real kwarg in current evalplus). Silent `except Exception: return False` made every problem look like a fail. Replaced with the subprocess-based pattern from `diffucoder_experiments/.../bench/eval_diverse_fastdllm.py`.

These are the kinds of bugs an acceptance gate exists to catch.

## Reference papers

- **dKV-Cache** — arxiv [2505.15781](https://arxiv.org/abs/2505.15781)
- **Fast-dLLM** — arxiv [2505.22618](https://arxiv.org/abs/2505.22618)
- **SlowFast Sampling** — arxiv [2506.10848](https://arxiv.org/abs/2506.10848)
- **Self-speculative decoding for diffusion** — arxiv [2510.04147](https://arxiv.org/abs/2510.04147) (Phase 2)
- **Sparse-dLLM** — arxiv [2508.02558](https://arxiv.org/abs/2508.02558) (Phase 4)

## Hardware

Tested on a single NVIDIA RTX 5090 (Blackwell, 32 GB) with PyTorch 2.12 nightly cu128. Should also work on A100/H100 with stable PyTorch + CUDA 12.4 (FlexAttention path), though FA3-on-Blackwell is unavailable so we use SDPA fallback.

## Cost

Phase-1 development: 0 GPU hours (all on laptop). Phase-1 acceptance gate: ~$5 of vast.ai RTX 5090 time across spike + benchmark + debug iterations.

---

Built in collaboration with Claude (Anthropic). All code authored by the developer; planning and pair-programming by Claude.
