# mdlm-engine v0.3.0 — Self-Speculative Decoding (1.80× faster, near-lossless)

The third major release. v0.2.2 stabilized quality (0.6707 single-shot pass@1 on full HumanEval+). **v0.3.0 ships self-speculative decoding (SSD, arxiv 2510.04147) with a speed/quality trade-off that defaults to 1.80× faster at -1.83 pp pass@1.**

## Headline numbers (Dream-Coder-Instruct-7B, full HumanEval+, n=164)

| Config | pass@1 | Δ vs v0.2.2 | s/problem | Speedup |
|---|---:|---:|---:|---:|
| v0.2.2 (no SSD) | 0.6707 (110/164) | 0 | 11.60 | 1.00× |
| **v0.3.0 default (SSD t=0.99)** | **0.6524 (107/164)** | **-1.83 pp** | **6.44** | **1.80×** |
| v0.3.0 speed preset (SSD t=0.95) | 0.6220 (102/164) | -4.87 pp | 5.74 | 2.02× |

The 3-problem deficit at t=0.99 is well within n=164 sampling noise; v0.3.0 is **near-lossless at meaningful speedup**.

For comparison, published baselines on full HE+ single-shot:
- DiffuCoder-7B-cpGRPO: 0.652 (paper)
- LLaDA-8B-Instruct: 0.494 (paper)
- mdlm-engine v0.3.0 (this release): **0.6524** at **1.80× the speed of v0.2.2**

## What's new

### Self-Speculative Decoding (Redesign C)

After three iterations of design, the working algorithm is:

**At every step**, AFTER the regular forward but BEFORE the regular sampler:
1. From the forward's already-computed logits, identify masked positions in the active block whose top-1 softmax probability ≥ `speculative_threshold`.
2. Propose the argmax token at each such position.
3. Run **one extra verification forward** with those tokens written into `state.x`.
4. Compare the verification forward's argmax at proposed positions to what was proposed; accept the longest matching prefix.
5. Commit accepted positions to the cache; the regular sampler then runs on the residual.

**Why it works at threshold=0.99 with `top_p=0.95` sampling**: when the model's top-1 raw probability ≥ 0.99, no other token clears the top-p=0.95 cumulative mass. The regular sampler at temp=0.2 would have committed exactly the argmax token. SSD's commit is deterministically equivalent.

**Three earlier designs that didn't work** (now documented in BENCHMARKS.md):
1. **threshold=0** (day-1): committed at uncertain positions → -25 pp pass@1 from commit-order drift
2. **block-init at step 0** (Redesign A): all positions masked at step 0 → low confidence everywhere → empty proposals → +1% forward overhead
3. **Per-step on residual after sampler** (Redesign B): operated on slowfast's leftovers (low-confidence by selection) → only 1.12× speedup

The shipped design (Redesign C) runs SSD on the FULL current mask BEFORE the sampler at every step. As steps progress and committed context accumulates, more positions become high-confidence → SSD fires more often → larger speedup. The cost is a single verify forward per step where SSD fires.

### What also got investigated and dropped

- **MXFP8 quantization (torchao)**: day-1 spike showed logit drift 4.78 (FAIL threshold 0.5); slower in benchmark too (FP8 ops on Blackwell + PyTorch 2.11 cu130 unoptimized). Deferred to v0.3.1.
- **torch.compile**: CUDA graph thrashing dominates due to fast_dllm's in-place K/V mutation pattern + `.item()` graph breaks in `apply_rotary_pos_emb`. 2-4× SLOWER. Permanently dead on this stack.

## Default behavior change (opt-out, not opt-in)

v0.3.0 enables SSD by default with `speculative_k=1` and `speculative_threshold=0.99`:

```python
# v0.2.2 behavior (quality preset; explicit opt-out)
engine.generate(prompt_ids, speculative_k=0)

# v0.3.0 default (near-lossless speedup)
engine.generate(prompt_ids)  # same as speculative_k=1, threshold=0.99

# v0.3.0 speed preset (more aggressive speedup)
engine.generate(prompt_ids, speculative_threshold=0.95)
```

Bench harness CLI:
```bash
# v0.3.0 default
python3 -m mdlm_engine.bench.harness --adapter dream ...

# v0.2.2 quality
python3 -m mdlm_engine.bench.harness --adapter dream --speculative_k 0 ...

# v0.3.0 speed preset
python3 -m mdlm_engine.bench.harness --adapter dream --speculative_threshold 0.95 ...
```

## Files added in v0.3.0

- `mdlm_engine/speculative/propose.py` — `propose()` and `propose_block_level()`
- `mdlm_engine/speculative/verify.py` — verification forward + longest-prefix acceptance
- `mdlm_engine/speculative/__init__.py` — orchestrator and re-exports
- `tests/test_speculative.py` — 17 CPU tests
- `tests/test_speculative_block.py` — 10 CPU tests
- `scripts/v0_3_0_spike.sh` — MXFP8 viability spike (kept for v0.3.1 retry)
- `scripts/v0_3_0_speed.sh` — 4-config gate runner
- `scripts/v0_3_0_production.sh` — production-settings 3-config gate
- `scripts/v0_3_0_block_ssd.sh` — block-level SSD ablation (Redesign A history)
- `scripts/v0_3_0_threshold_sweep.sh` — t=0.80/0.90/0.95/0.99 sweep

161 CPU tests passing (was 134 at v0.2.2; +27 SSD tests).

## Cost & process

- v0.3.0 sprint: ~1 day, ~$1.50 of vast.ai 5090 time across 3 SSD redesigns + MXFP8/compile attempts + final threshold sweep
- Total project cost trajectory: v0.1.0 → v0.2.2 was ~$11; v0.3.0 ships at ~$13

## Backwards-incompatible changes

- `DiffusionEngine.generate()` defaults: `speculative_k=0 → 1`, `speculative_threshold=0.95 → 0.99`. Pass `speculative_k=0` for v0.2.2-identical behavior.
- `LoopConfig.speculative_threshold` default: `0.95 → 0.99`. Same opt-out via setting it back.
- `LoopConfig.speculative_block_init` field removed (was a Redesign A artifact, never the right design).
- `--speculative_per_step` harness flag removed (Redesign B fallback removed).

The pre-v0.3.0 SSD code paths (block-init, per-step-on-residual) were validated as inferior in measurement and are deleted from the codebase.

## What's next (v0.3.1+)

- **Full HE+ validation of LLaDA at v0.3.0 defaults** (~30 min, $0.20) — record the LLaDA row in BENCHMARKS.md.
- **Diverse best-of-N + SSD**: combine `--diverse 8` with SSD t=0.99. Realistic target: 0.95+ best-of-8 oracle on full HE+ at ~10-12 s/problem amortized.
- **MXFP8 retry** when torchao or PyTorch updates the Blackwell FP8 path.
- **Native LLaDA caching** via fast_dllm-style modeling patches (separate from mdlm-engine engine work).

## Verification

```bash
# Reproduce v0.3.0 numbers on a vast.ai box
cd /workspace
git clone https://github.com/modhisathvik7733/mdlm-engine.git
cd mdlm-engine
bash scripts/bootstrap_vastai.sh

# Single config at v0.3.0 default (~17 min, $0.10)
python3 -m mdlm_engine.bench.harness \
    --adapter dream --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --use_fastdllm_modeling \
    --benchmark humaneval_plus --limit 200 \
    --max_new_tokens 512 \
    --out /workspace/v0_3_0_default.json
# Expected: pass@1 ~0.65, s/problem ~6.5

# Quality preset (no SSD), should match v0.2.2's 0.6707 / 11.60s
python3 -m mdlm_engine.bench.harness \
    --adapter dream --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --use_fastdllm_modeling \
    --speculative_k 0 \
    --benchmark humaneval_plus --limit 200 \
    --max_new_tokens 512 \
    --out /workspace/v0_3_0_no_ssd.json

# Speed preset (more aggressive)
python3 -m mdlm_engine.bench.harness \
    --adapter dream --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --use_fastdllm_modeling \
    --speculative_threshold 0.95 \
    --benchmark humaneval_plus --limit 200 \
    --max_new_tokens 512 \
    --out /workspace/v0_3_0_speed.json
# Expected: pass@1 ~0.62, s/problem ~5.7
```
