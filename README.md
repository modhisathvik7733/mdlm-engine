# mdlm-engine

A **model-agnostic Python inference engine for masked diffusion language models** — Dream-Coder, DiffuCoder, LLaDA, and any future masked diffusion LM that fits the adapter contract.

> **Status: Phase 1 code-complete (Days 1-7).** Acceptance gate (Days 8-9) requires a GPU run; see [`scripts/phase1_acceptance.sh`](scripts/phase1_acceptance.sh). 92 unit tests passing on CPU.

## Why this exists

Existing inference accelerators for masked diffusion LMs (`fast_dllm`, etc.) are **hard forks of one model's modeling code**. They give real speedups, but porting to a new model means re-doing all of the modeling-code surgery — RoPE, position-id rewrites, custom attention masks, cache integration. That's a non-starter for a research artifact that needs to outlive a single model release.

`mdlm-engine` integrates **only at HF transformers' public boundaries** (`AutoModel.forward()` and the `Cache` API), with model-specific facts isolated in a small `ModelAdapter` subclass per model family. Adding a new diffusion LM is targeted at **≤ 180 lines of Python**.

## Headline contribution

```
Same engine code runs Dream-Coder / DiffuCoder / LLaDA / (your model here),
with ~150 LOC of model-specific adapter, zero changes to generation, cache,
or scheduler logic.
```

Reportable portability metrics (target at v0.1.0):
- **3 model families** supported out of the box
- **≤ 180 LOC** median per-model adapter
- **≤ 2 hours** from `git clone` to first generation on a new model
- **Zero per-model branches** in core engine code

Speed is the *validation* that portability didn't cost us anything — not the headline.

## What's in v0.1.0 (Phase 1, in development)

- `DiffusionEngine` — the model-agnostic generation entrypoint.
- `ModelAdapter` ABC + adapters for Dream-Coder and LLaDA.
- `DiffusionCache` ABC + `BlockCache` (naive prefix) and `DKVCache` (delayed, position-indexed; arxiv [2505.15781](https://arxiv.org/abs/2505.15781)).
- Schedulers: uniform, confidence-threshold, slowfast (arxiv [2506.10848](https://arxiv.org/abs/2506.10848)).
- Samplers: argmax, entropy, margin, topk_margin, maskgit_plus.
- Optional Blackwell hardware opts: MXFP8 via `torchao`, `torch.compile(reduce-overhead, fullgraph=True)`, FlexAttention `mask_mod` for bidirectional attention.
- Adapter validation contract (4 mandatory tests; refusal to register if any fail).
- Bench harness reporting: s/problem, tokens/sec, single-shot pass@1, best-of-N pass@1, peak VRAM.

## Roadmap

| Version | Phase | Theme | Target |
|---|---|---|---|
| **v0.1.0** | Phase 1 | dKV-Cache + slowfast scheduler + Blackwell HW opts + 2 adapters | fast_dllm parity (≈ 8 s/problem on Dream HE+ subset) at 0.55 single-shot pass@1 |
| v0.2.0 | Phase 2 | Self-speculative decoding (arxiv [2510.04147](https://arxiv.org/abs/2510.04147)) | ≤ 2.5 s/problem, ≥ 0.62 single-shot pass@1 |
| v0.3.0 | Phase 3 | Continuous batching | ≥ 3× throughput at batch 8 |
| v0.4.0 | Phase 4 | Sparse cache eviction (arxiv [2508.02558](https://arxiv.org/abs/2508.02558)) | long-context wins; 1.4× larger feasible batch at same VRAM |

See [`/Users/chintu/.claude/plans/jazzy-tickling-brook.md`](file:///Users/chintu/.claude/plans/jazzy-tickling-brook.md) for the full plan.

## Running the Phase-1 acceptance gate

On a vast.ai (or similar) GPU box with CUDA + PyTorch nightly:

```bash
git clone https://github.com/modhisathvik7733/mdlm-engine.git
cd mdlm-engine
pip install --break-system-packages -e ".[bench,test]"

# Optional: run the three Day-1 spike scripts first (architecture-freeze evidence,
# already committed to scripts/day1_spike/*.json):
bash scripts/day1_spike/00_setup_fast_dllm.sh   # if not already done
python3 scripts/day1_spike/01_llada_spike.py --llada_path GSAI-ML/LLaDA-8B-Base
python3 scripts/day1_spike/02_host_overhead_profile.py --model_path /workspace/models/dream-coder-7b-instruct
python3 scripts/day1_spike/03_mxfp8_viability.py --model_path /workspace/models/dream-coder-7b-instruct

# The actual Phase-1 acceptance gate (~2-3 hours of GPU time):
bash scripts/phase1_acceptance.sh
```

The gate runs:

1. `pytest tests/` — all green (92 CPU + 2 GPU contract tests).
2. Dream-Coder benchmark on HumanEval+ subset, single-shot AND best-of-8.
3. LLaDA portability smoke (same engine code, different adapter).
4. Adapter LOC check (each ≤ 200 LOC).
5. NaN-freedom (100 generations at temperature 0, zero NaN logits).

Acceptance thresholds (from the v2 plan):

```
Dream single-shot pass@1  ≥ 0.55     speed ≤ 8 s/problem (≈ fast_dllm parity)
Dream best-of-8 pass@1    ≥ 0.85
LLaDA smoke pass@1        ≥ 0.40
NaN count                  = 0
Adapter LOC each          ≤ 200
```

If all green → tag `v0.1.0` and ship.

## Honest non-goals

- Custom CUDA kernels.
- Distillation / consistency distillation (those produce model-specific artifacts; out of scope).
- Multi-GPU / tensor-parallel inference.
- Wrapping or vendoring `vLLM` / `fast_dllm`.

## Reference papers

1. **dKV-Cache** — arxiv [2505.15781](https://arxiv.org/abs/2505.15781). Phase 1 cache.
2. **Fast-dLLM** — arxiv [2505.22618](https://arxiv.org/abs/2505.22618). Reference for parallel decoding heuristics.
3. **SlowFast Sampling** — arxiv [2506.10848](https://arxiv.org/abs/2506.10848). Phase 1 scheduler.
4. **Self-Speculative Decoding for Diffusion** — arxiv [2510.04147](https://arxiv.org/abs/2510.04147). Phase 2 core.
5. **Sparse-dLLM** — arxiv [2508.02558](https://arxiv.org/abs/2508.02558). Phase 4.

Plus published baselines we hold ourselves to:
- **DiffuCoder** — arxiv [2506.20639](https://arxiv.org/abs/2506.20639). Single-shot HE+ pass@1: 65.2%.
- **LLaDA** — arxiv [2502.09992](https://arxiv.org/abs/2502.09992). Single-shot HE+ pass@1: 49.4%.

## License

MIT. See `LICENSE`.
