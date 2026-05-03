# mdlm-engine — Benchmark History

Canonical results log for every tagged release. Each row is a real measurement on the listed hardware; no projections, no simulations. v0.2.2 is the first version with full HumanEval+ (n=164) data — earlier versions used n=20 subset only.

## Headline table — Dream-Coder-v0-Instruct-7B on HumanEval+ single-shot

| Tag | Date | n | max_new | s/problem | tokens/sec | pass@1 single-shot | Path |
|---|---|---:|---:|---:|---:|---:|---|
| v0.1.0 | 2026-05-02 | 20 | 256 | 9.9 | 22 | 0.900 (18/20) | C (no caching) |
| v0.2.0 | 2026-05-03 | 20 | 256 | 11.31 | 19.5 | 0.950 (19/20) | C (PATH A code shipped but model on B/C) |
| v0.2.0 | 2026-05-03 | **164** | 256 | 12.70 | 18.5 | **0.5366 (88/164)** | C (full HE+) |
| v0.2.1 | 2026-05-03 | 20 | 256 | 8.42 | 25.5 | 0.850 (17/20) | A (fast_dllm-patched modeling) |
| v0.2.1 | 2026-05-03 | **164** | 256 | 9.14 | 25.3 | **0.5427 (89/164)** | A (full HE+) |
| **v0.2.2** | **2026-05-03** | **164** | **512** | **11.60** | **25.1** | **0.6707 (110/164)** | **A (default)** |
| v0.3.0 candidate (no SSD) | 2026-05-04 | 20 | 512 | 9.82 | 25.4 | 0.800 (16/20) | A |
| v0.3.0 candidate (SSD t=0.95) | 2026-05-04 | 20 | 512 | 4.19 | 52.7 | 0.800 (16/20) | A + SSD Redesign C |
| v0.3.0 (SSD t=0.90, drift) | 2026-05-04 | 20 | 512 | 3.93 | 57.8 | 0.600 (12/20) | A + SSD (too aggressive) |
| **v0.3.0 (SSD t=0.95) full HE+** | **2026-05-04** | **164** | **512** | **5.74** | **51.2** | **0.6220 (102/164)** | **A + SSD Redesign C** |

The n=20 numbers (0.900-0.950) inflated because the first 20 HE+ problems are easy and fit in max_new=256. **The honest baseline is 0.6707 on full HE+ at v0.2.2's defaults.** Slightly above DiffuCoder-7B-cpGRPO's 0.652 published number.

**v0.3.0 SSD t=0.95 vs v0.2.2 baseline (n=164, both at PATH A 512, temp=0.2, entropy, top_p=0.95):**
- pass@1: 0.6707 → **0.6220** = **-4.87 pp** (real but small quality cost)
- s/problem: 11.60 → **5.74** = **2.02× faster**
- forwards: 38170 → **22791** = **-40%**
- VRAM: 16.17 GB → 16.29 GB (unchanged)

The n=20 "0.80 = 0.80" identity at this config was sample-size noise — full HE+ shows there IS a real ~5 pp pass@1 cost from SSD's argmax-vs-argmax verify diverging slightly from the regular sampler's stochastic outputs at temp=0.2. Trade-off is real but modest: ~2× speedup costs ~5 pp pass@1.

**Comparison with published baselines on full HE+ single-shot:**
- mdlm-engine v0.2.2 (no SSD): 0.6707 — slightly above DiffuCoder
- **mdlm-engine v0.3.0 (SSD t=0.95): 0.6220** — slightly below DiffuCoder (0.652) at 2× the speed
- DiffuCoder-7B-cpGRPO (paper, no engine info): 0.652
- LLaDA-8B-Instruct (paper): 0.494

## Path / cache-wiring summary

- **PATH A**: fast_dllm-patched `modeling_dream.py` (bundled in `mdlm_engine/models/dream_fastdllm/`). `dual_cache=True`, `replace_position` true at active block, K/V reused across iter steps. ~1.4× faster than PATH C.
- **PATH B**: stock HF caching — accurately detected but **collapsed to PATH C in adapter** because stock HF caching is append-only and cannot accelerate masked diffusion.
- **PATH C**: no caching, full forward each step. v0.1.0 behavior, baseline.

## v0.2.x investigation summary (what took us to 0.67 from 0.54)

Three false hypotheses before landing on the right one:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | K/V drift across cached steps | `steps_per_block=16` ablation | ~0pp net recovery; drift real but recovery washes. NOT the cause. |
| 2 | Engine quality regression vs native | Native `model.diffusion_generate()` on same prompts | REVERSED — mdlm-engine BEATS native by 5-20 pp at every checkpoint. Engine validated. |
| 3 | Generation budget (max_new=256 truncating) | Bump to max_new=512 | 67 truncated-fail → 29 recover + 31 other gains = **+12.8 pp pass@1**. ✓ |

**Lesson**: budget was the bottleneck, not the engine. Subsets inflated v0.1.0/v0.2.0/v0.2.1 release-notes pass@1 because easy-problem subset fit in 256 tokens.

## v0.3.0 sprint summary (what didn't work and what did)

Three speed levers attempted; only one paid off:

| Lever | Day | Result | Verdict |
|---|---|---|---|
| **MXFP8** (torchao quantization) | 2 | day-1 spike: logit drift 4.78 (FAIL threshold 0.5); -20pp pass@1; **slower** in benchmark (FP8 ops on Blackwell + PyTorch 2.11 cu130 unoptimized) | dead, defer to v0.3.1 |
| **torch.compile** (mode=default + cudagraph_support_input_mutation) | 2 | CUDA graph thrashing dominates (Inductor's `cudagraph_support_input_mutation=True` doesn't fix `.item()` graph breaks in apply_rotary_pos_emb); 2-4× SLOWER | dead |
| **Self-Speculative Decoding** (3 redesigns) | 2-4 | Redesigns A & B (per-step on residual / block-init at step 0) gave only 1.04-1.12×. Redesign C (per-step on FULL mask BEFORE sampler at every step) gave **2.34× lossless** | **shipped** |

### SSD Redesign C — why it works

At `temp=0.2, top_p=0.95` production sampling, when the model's top-1 probability ≥ 0.95, top-p sampling has no other candidates → sampler picks the argmax deterministically. SSD's threshold=0.95 commit at exactly those positions matches what the sampler would have committed. **Lossless within sampling noise.**

Three earlier SSD designs failed by fighting this property:
1. **threshold=0** (day-1): committed at uncertain positions → 25 pp drop from commit-order drift.
2. **block-init at step 0** (Redesign A): all positions masked at step 0 → low confidence everywhere → empty proposals → +1% forwards (verify overhead).
3. **Per-step on residual after sampler** (Redesign B): operated on slowfast's leftovers → low-confidence positions → only 1.12× speedup.

Redesign C runs SSD BEFORE the sampler at every step on the FULL current mask. Adapts to problem difficulty: boilerplate-heavy problems (many high-conf positions) → big speedup; logic-heavy problems (few high-conf positions) → SSD fires rarely, near-baseline speed but quality preserved.

### v0.3.0 candidate breakdown — full HE+ (n=164, 2026-05-04)

| metric | v0.2.2 baseline (recorded) | v0.3.0 SSD t=0.95 (measured) | Δ |
|---|---:|---:|---:|
| Settings | PATH A 512, temp=0.2, entropy, top_p=0.95 | + speculative_k=1, threshold=0.95 | (SSD added) |
| pass@1 | **0.6707** (110/164) | **0.6220** (102/164) | **-4.87 pp** |
| s/problem | 11.60 | **5.74** | **2.02× faster** |
| tokens/sec | 25.1 | **51.2** | **+104%** |
| total forwards | 38170 | 22791 | **-40%** |
| peak VRAM (GB) | 16.17 | 16.29 | unchanged |

The 2× speedup is mechanical (forward count drops 40%, wall-clock follows). The 4.87 pp pass@1 cost is the price of approximating temp=0.2 sampling with argmax at high-confidence positions — at n=164 the imperfect approximation accumulates to ~8 problem-level disagreements (102 pass vs 110 pass).

This is a real **speed/quality trade-off offering**, not a strict improvement:
- v0.2.2 default: quality preset (0.6707 / 11.60s)
- v0.3.0 SSD t=0.95: speed preset (0.6220 / 5.74s)

n=20 sample variance hid this gap (showed 0.80 = 0.80 identity); only full HE+ revealed the real cost.

## LLaDA-8B-Base (portability target)

| Tag | n | max_new | s/problem | pass@1 | Path |
|---|---:|---:|---:|---:|---|
| v0.1.0 | 20 | 256 | 12.6 | 0.500 | C |
| v0.2.0 | 20 | 256 | 14.44 | 0.500 | C |
| v0.2.1 | 20 | 256 | 13.72 | 0.500 | C |
| v0.2.2 | TBD | 512 | TBD | TBD | C (caching deferred) |

LLaDA's modeling lacks `dual_cache`/`replace_position`. PATH A unavailable; falls through to PATH C cleanly with a one-time warning. End-to-end runs unchanged across v0.2.x.

## Cache equivalence (correctness gate)

At every release ≥ v0.2.0:
- **block ≡ dkv** identical token-by-token at temperature 0 across 5 prompts
- 0 NaN logits in 100 successive generations at temp 0

## Compute cost per release

| Phase | Days | Vast.ai $5090 hours | $ |
|---|---:|---:|---:|
| Phase 1 (v0.1.0) | 3 | ~10 | ~$5 |
| Phase 2.0 (v0.2.0) | 5 | ~10 | ~$3 |
| Phase 2.1 (v0.2.1) | 1 | ~3 | ~$1 |
| Phase 2.2 (v0.2.2 investigation) | 1 | ~5 | ~$2 |
| Phase 3.0 (v0.3.0 sprint — 3 SSD redesigns + MXFP8 spike + compile retry) | 1 | ~4 | ~$1.50 |
| **Total to v0.3.0 candidate** | **11** | **~32** | **~$12.50** |

## Reproducing these numbers

```bash
# Bootstrap a fresh vast.ai box (image: vastai/pytorch_cuda-13.0.2-auto)
cd /workspace
git clone https://github.com/modhisathvik7733/mdlm-engine.git
cd mdlm-engine
bash scripts/bootstrap_vastai.sh

# Reproduce v0.2.2 acceptance gate (~80 min wall, ~$0.50)
bash scripts/phase2_1_acceptance.sh

# Reproduce v0.3.0 candidate (n=20, ~7 min, ~$0.05)
bash scripts/v0_3_0_production.sh

# Reproduce v0.3.0 candidate at full HE+ (~25 min, ~$0.18)
LIMIT=200 bash scripts/v0_3_0_production.sh

# Or run a single config on full HumanEval+ with v0.3.0 SSD defaults
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --use_fastdllm_modeling \
    --max_new_tokens 512 \
    --speculative_k 1 \
    --speculative_threshold 0.95 \
    --limit 200 \
    --out /workspace/v0_3_0_pathA_ssd.json
```

## v0.3.1+ — backlog

- **Full HE+ validation of v0.3.0 SSD** (~25 min, ~$0.18) — pending; expected pass@1 ~0.67 / s/problem ~5.
- **Diverse best-of-N**: implement `--diverse 8` integration with SSD. Expected: ~0.95+ best-of-8 oracle on full HE+ at ~8-10 s/problem amortized.
- **MXFP8 retry** when torchao or PyTorch updates Blackwell FP8 path (current logit drift 4.78 is unusable).
- **Native LLaDA caching** (separate work): requires patched LLaDA modeling with `dual_cache`/`replace_position` extensions.
- **Sparse-dLLM** (arxiv 2508.02558): 5.8× claimed at long context. HE+ prompts too short for the win to show; useful for LiveCodeBench / multi-turn.

---

This file is the durable record of measured numbers. Release notes (`RELEASE_NOTES_v0.X.Y.md`) cover what changed in each release; this file tracks the running scoreboard.
