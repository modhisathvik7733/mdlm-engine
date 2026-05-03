#!/usr/bin/env bash
# v0.2.3 quick test: does torch.compile actually speed up PATH A on Blackwell?
#
# Runs n=20 (NOT full HE+) twice — once without --compile, once with —
# at v0.2.2 defaults (PATH A 512). Same prompts, so the s/problem ratio
# is a clean compile speedup measurement. Quality should be identical
# (or warn if pass@1 differs, which would indicate a compile-induced bug).
#
# Cost: ~12 min wall (~$0.08).

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_2_3_compile"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.2.3 quick test — torch.compile on PATH A 512 (n=$LIMIT)"
echo "============================================================"
echo

echo "[1/3] Baseline: PATH A 512, no compile (control)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/pathA_nocompile.json" 2>&1 | \
    tee "$RESULTS_DIR/pathA_nocompile.log"
echo

echo "[2/3] PATH A 512 + --compile (torch.compile reduce-overhead)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --compile \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/pathA_compile.json" 2>&1 | \
    tee "$RESULTS_DIR/pathA_compile.log"
echo

echo "[3/3] Compile speedup analysis"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/compile_diff.log"
import json
import os

rd = os.environ["RESULTS_DIR"]
with open(f"{rd}/pathA_nocompile.json") as f: nc = json.load(f)
with open(f"{rd}/pathA_compile.json")   as f: c  = json.load(f)

print()
print(f"{'metric':25s}  {'no compile':>12s}  {'+ compile':>12s}  {'delta':>10s}")
print("-" * 65)
print(f"{'pass@1':25s}  {nc['pass_at_1_single_shot']:>12.4f}  {c['pass_at_1_single_shot']:>12.4f}  {(c['pass_at_1_single_shot']-nc['pass_at_1_single_shot'])*100:>+8.1f}pp")
print(f"{'s/problem':25s}  {nc['seconds_per_problem']:>12.2f}  {c['seconds_per_problem']:>12.2f}  {nc['seconds_per_problem']/c['seconds_per_problem']:>9.2f}x")
print(f"{'tokens/sec':25s}  {nc['tokens_per_second']:>12.1f}  {c['tokens_per_second']:>12.1f}  {c['tokens_per_second']/nc['tokens_per_second']:>9.2f}x")
print(f"{'wall (s)':25s}  {nc['wall_seconds']:>12.0f}  {c['wall_seconds']:>12.0f}")
print(f"{'peak VRAM (GB)':25s}  {nc['peak_vram_gb']:>12.2f}  {c['peak_vram_gb']:>12.2f}")
print()

speedup = nc['seconds_per_problem'] / c['seconds_per_problem']
quality_delta = (c['pass_at_1_single_shot'] - nc['pass_at_1_single_shot']) * 100
print("=" * 65)
if speedup >= 1.20:
    print(f"VERDICT: torch.compile gives {speedup:.2f}x speedup. Worth shipping.")
    print(f"  Action: make --compile the v0.2.3 default if quality holds (delta {quality_delta:+.1f}pp).")
elif speedup >= 1.05:
    print(f"VERDICT: marginal speedup ({speedup:.2f}x). Probably not worth the recompile time.")
    print(f"  Note: first n problems include compile warmup; longer runs would help.")
else:
    print(f"VERDICT: torch.compile didn't help on this stack ({speedup:.2f}x).")
    print(f"  Likely Blackwell + nightly compile fallback or graph break. Move on to other levers.")

if abs(quality_delta) > 3:
    print(f"  WARNING: pass@1 drifted by {quality_delta:+.1f}pp — compile may have broken numerics.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
