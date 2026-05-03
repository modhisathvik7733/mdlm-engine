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

The n=20 numbers (0.900-0.950) inflated because the first 20 HE+ problems are easy and fit in max_new=256. **The honest baseline is 0.6707 on full HE+ at v0.2.2's defaults.** Slightly above DiffuCoder-7B-cpGRPO's 0.652 published number.

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
| **Total to v0.2.2** | **10** | **~28** | **~$11** |

## Reproducing these numbers

```bash
# Bootstrap a fresh vast.ai box (image: vastai/pytorch_cuda-13.0.2-auto)
cd /workspace
git clone https://github.com/modhisathvik7733/mdlm-engine.git
cd mdlm-engine
bash scripts/bootstrap_vastai.sh

# Reproduce v0.2.2 acceptance gate (~80 min wall, ~$0.50)
bash scripts/phase2_1_acceptance.sh

# Or run a single config on full HumanEval+ (~30-50 min depending on max_new)
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --use_fastdllm_modeling \
    --max_new_tokens 512 \
    --limit 200 \
    --out /workspace/v0_2_2_pathA_512.json
```

## v0.3.0 (next) — backlog

- **Diverse best-of-N**: implement `--diverse 8`. Expected: ≥0.95 best-of-8 oracle on full HE+ (single-shot 0.6707 → `1 − (1−0.67)^8 ≈ 0.999`).
- **Self-speculative decoding** (arxiv 2510.04147): generate K candidates in parallel, verify in one forward, accept longest verified prefix. Realistic target: 5-7 s/problem at 0.65+ pass@1.
- **Native LLaDA caching** (separate work): requires patched LLaDA modeling with `dual_cache`/`replace_position` extensions.

---

This file is the durable record of measured numbers. Release notes (`RELEASE_NOTES_v0.X.Y.md`) cover what changed in each release; this file tracks the running scoreboard.
