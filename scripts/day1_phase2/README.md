# Phase 2 Day-1 Spike

Single script that **gates Phase 2's architecture**. ~30 seconds on a GPU box.

| Script | Time | Question |
|---|---|---|
| `verify_dual_cache.py` | ~30 sec | Does the HF Hub Dream-Coder model accept fast_dllm's `dual_cache` and `replace_position` kwargs? |

## How to run

```bash
cd /workspace/mdlm-engine
git pull origin main
python3 scripts/day1_phase2/verify_dual_cache.py \
    --dream_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --llada_path GSAI-ML/LLaDA-8B-Base
```

## Decision rules from the output

The script writes `dual_cache_support.json` and prints one of three verdicts.

### PATH A — Full dual_cache support
Dream forward accepts `dual_cache`, `replace_position`, `past_key_values`, `use_cache`.

→ **Phase 2 Dream wiring is the simplest path**: `model.forward(past_key_values=cache.to_legacy_kv(), dual_cache=True, replace_position=~commit_state, use_cache=True)`. Expected ~2× speedup over v0.1.0.

### PATH B — `past_key_values` supported, `dual_cache` missing
Dream accepts standard HF caching kwargs but not fast_dllm's extensions.

→ **Phase 2 ships with concat-only cache updates** (no in-place K/V replacement at masked positions). Speedup will be smaller (~1.3-1.5× over v0.1.0). Document this as a v0.2.0 known limitation; v0.2.1 could add in-place support if we choose to fork the modeling code.

### PATH C — Neither supported
Dream's stock HF forward doesn't accept past_key_values at all.

→ **Phase 2 is blocked on Dream.** Three sub-options:
1. Document that mdlm-engine v0.2.0 requires the fast_dllm-patched Dream-Coder modeling files.
2. Ship Phase 2 with LLaDA caching only (if it supports past_key_values), keep Dream at v0.1.0 speed.
3. Defer Phase 2 entirely; revisit when stock HF Dream-Coder gains caching support.

## After the spike

Commit the JSON to the repo (without weights — only the signature findings) so the architecture decision is reproducible:

```bash
git add scripts/day1_phase2/dual_cache_support.json
git commit -m "Phase 2 Day-1 spike: dual_cache support verdict for Dream + LLaDA"
git push origin main
```
