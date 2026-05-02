# mdlm-engine

A **model-agnostic Python inference engine for masked diffusion language models** — Dream-Coder, DiffuCoder, LLaDA, and any future masked diffusion LM that fits the adapter contract.

> **Status: Pre-alpha.** Phase 1 in development. Not yet shippable.

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
