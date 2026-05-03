# mdlm-engine v0.2.0 — cache-wiring infrastructure

A focused infrastructure release. v0.1.0 kept `DiffusionCache` as engine-side bookkeeping only — the adapter passed `use_cache=False` to the model, so K/V was recomputed every diffusion step. **v0.2.0 builds the bridge** between the engine's cache and the model's `past_key_values` arg, but the actual speedup is gated on the model's modeling code exposing fast_dllm's `dual_cache` + `replace_position` extensions, which the upstream HF Hub Dream-Coder model **does not**. Tagging v0.2.0 ships the bridge; **v0.2.1 will bundle a fast_dllm-patched `modeling_dream.py`** to engage the speedup.

## What's new

- **`DiffusionCache.to_legacy_kv()`**: returns `[(K, V), ...]` per layer as aliases of the cache's internal `_K`/`_V`. The model can write through these in-place; cache reads see the writes automatically. Default impl on the ABC; `BlockCache` and `DKVCache` inherit it; `NoOpCache` overrides to return `[None] * n_layers` (the explicit "no cache" signal HF expects).
- **`DKVCache.update_from_model_output()`**: bypasses the strict committed-slot check (the model's returned K/V is authoritative even at frozen positions; only the *engine* should be tripwire'd by re-asking for recompute).
- **`DreamAdapter.forward()` PATH dispatch**: detects which kwargs `model.forward` accepts via `inspect.signature` at construction time, then routes one of three ways:
  - **PATH A** — full fast_dllm extensions (`dual_cache=True`, `replace_position=~commit_state`). In-place K/V replace at masked positions, ~2× speedup. Currently unreachable on the HF Hub model.
  - **PATH B** — stock HF caching (`past_key_values + use_cache` only). Detected accurately but **collapsed to PATH C in `forward()`** with a one-time `RuntimeWarning`. Stock HF caching is append-only (`torch.cat([past, new], dim=-2)`) and cannot accelerate masked diffusion (which passes the full sequence each step and needs in-place replace at masked positions).
  - **PATH C** — no caching support. v0.1.0 behavior with `use_cache=False`.
- **Loop integration**: `core/loop.py` now requests `use_cache=True`. The adapter handles the dispatch.
- **LLaDA caching deferred to v0.2.1**: LLaDA's modeling lacks `dual_cache`/`replace_position` (verified `scripts/day1_spike/01_llada_spike.json`), so the LLaDA adapter explicitly keeps `use_cache=False`. LLaDA still runs end-to-end at v0.1.0 speed; this is *not* a portability regression, just a deferred speedup target.
- **`scripts/day1_phase2/verify_dual_cache.py`**: the spike that surfaces a model's caching path before any engine code runs.
- **`scripts/cache_equivalence.py`** and **`scripts/phase2_acceptance.sh`**: the new gate.

## Acceptance results (HumanEval+, limit=20, RTX 5090, 2026-05-03)

| Model | pass@1 single-shot | s/problem | tokens/sec | Peak VRAM | NaN / 100 |
|---|---:|---:|---:|---:|---:|
| `Dream-org/Dream-Coder-v0-Instruct-7B` (PATH B → C) | **0.950** (19/20) | 11.92 | 18.7 | 22.0 GB | 0 |
| `GSAI-ML/LLaDA-8B-Base` (PATH C, deferred) | **0.500** (10/20) | 14.44 | 17.7 | 23.1 GB | — |

Plus:
- **117 CPU unit tests** passing (was 92 in v0.1.0; +25 across `test_cache_to_legacy_kv.py`, `test_cache_update_from_model.py`, `test_dream_adapter_caps.py`).
- **2 GPU-marked `adapter_validation` tests** now active (were skipped pending Phase-2 wiring).
- **Cache equivalence** (NEW gate): `none ≡ block ≡ dkv` — identical output on 5 prompts × 3 cache pairs at temperature 0.

Hardware parity vs v0.1.0's recorded numbers (different vast.ai box; ±20% s/problem variance is normal); pass@1 unchanged or improved.

## Why no speedup yet

The upstream HF Hub `Dream-org/Dream-Coder-v0-Instruct-7B` exposes a stock HF forward signature: `past_key_values` and `use_cache` are present, but `dual_cache` and `replace_position` are not. Stock HF caching is append-only — it concatenates new K/V onto the past tensor — which is fundamentally incompatible with masked diffusion's per-step in-place replace at masked positions. The adapter detects this and falls back cleanly with a `RuntimeWarning` rather than calling into a broken code path.

The PATH A code is **fully implemented and unit-tested** (`tests/test_dream_adapter_caps.py::test_caps_detect_path_a_full_fast_dllm`). It is exercised the moment the model's modeling code exposes the right kwargs. v0.2.1 will bundle a fast_dllm-patched `modeling_dream.py` so that exact loading path becomes the default.

## Backwards-incompatible changes

None.

## What this enables for v0.2.1

1. Bundle `mdlm_engine/models/dream_fastdllm/modeling_dream.py` (the fast_dllm-patched modeling).
2. Document `AutoModel.from_pretrained` from that local path.
3. Re-run the same `phase2_acceptance.sh` — the spike will print PATH A, the adapter will engage `dual_cache=True`, and the gate's Dream s/problem should drop from ~12 → ~5 with pass@1 unchanged.

The tagged release also intentionally separates "did the cache wiring code work?" (v0.2.0, **yes** — see the green gate) from "did the speedup engage on a particular model?" (v0.2.1, separate question).

## Cost

Phase-2 development: ~$2 of vast.ai RTX 5090 time (one bootstrap + one acceptance run).

## Hardware

Tested on a single NVIDIA RTX 5090 (Blackwell, 32 GB) with PyTorch 2.11 cu130. Should also work on A100/H100 with stable PyTorch + CUDA 12.4.

---

Built in collaboration with Claude (Anthropic). All code authored by the developer; planning and pair-programming by Claude.
