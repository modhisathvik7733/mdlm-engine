# Phase 1 Day-1 Spike

Three short scripts that **gate the rest of Phase 1**. Total ~5 hours of GPU work; their output drives architecture decisions.

| # | Script | Time | Question answered | Output |
|---|---|---|---|---|
| 1 | `01_llada_spike.py` | ~2 h | Does LLaDA fit our `ModelAdapter` ABC, or do we need to revise it? | `01_llada_spike.json` |
| 2 | `02_host_overhead_profile.py` | ~2 h | What fraction of wall time is Python/host overhead vs GPU? | `02_host_overhead_profile.json` |
| 3 | `03_mxfp8_viability.py` | ~1 h | Does MXFP8 quantization survive the diffusion sampling loop without NaN? | `03_mxfp8_viability.json` |

## How to run (vast.ai box with Dream-Coder + LLaDA downloaded)

```bash
cd /workspace
git clone https://github.com/<user>/mdlm-engine.git
cd mdlm-engine
pip install --break-system-packages -e ".[quant,bench]"

# Run all three (about 5 hours total on RTX 5090):
python scripts/day1_spike/01_llada_spike.py --llada_path GSAI-ML/LLaDA-8B-Base
python scripts/day1_spike/02_host_overhead_profile.py --model_path /workspace/models/dream-coder-7b-instruct
python scripts/day1_spike/03_mxfp8_viability.py --model_path /workspace/models/dream-coder-7b-instruct
```

The JSON outputs commit to the repo (without weights — see `.gitignore`) so the architecture decisions are reproducible.

## Decision rules from the outputs

### From `01_llada_spike.json`
- If `forward_signature_matches_abc: true` and `eos_token_ids` known → ABC stays as drafted.
- If LLaDA needs an extra method we don't have (e.g. an extra preprocessing hook) → revise `ModelAdapter` ABC **before** writing the engine.

### From `02_host_overhead_profile.json`
- If `host_fraction_pct > 30` → `torch.compile(mode="reduce-overhead")` with static padding is **load-bearing**, not optional, in Phase 1.
- If `host_fraction_pct ≤ 30` → torch.compile is a Phase-1 nice-to-have, not load-bearing.

### From `03_mxfp8_viability.json`
- If `max_abs_logit_diff < 0.05` → MXFP8 ships in v0.1.0.
- If `max_abs_logit_diff ∈ [0.05, 0.5]` → MXFP8 ships behind a `--quant mxfp8` flag, default off, with a "may NaN at low temperature" warning.
- If `max_abs_logit_diff > 0.5` or `nan_count > 0` → defer MXFP8 to Phase 2.
