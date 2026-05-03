#!/usr/bin/env bash
# Validate SSD's lossless property at temperature=0 (n=20, ~5 min, $0.04).
#
# Yesterday's full gate run showed SSD at temp=0.2 gave 1.94× speedup but
# -25pp pass@1. That's expected: SSD's argmax-vs-argmax acceptance test is
# only consistent with the engine's per-step output distribution when the
# regular sampler also uses argmax (i.e., temperature=0).
#
# This script runs two configs at temp=0:
#   1. PATH A 512 baseline, no SSD
#   2. PATH A 512 + SSD k=4
# If pass@1 matches within ~3pp, SSD is verified lossless and we ship
# v0.3.0 with "SSD k=4 + temp=0" as the recommended speed mode.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_3_0_ssd_temp0"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.3.0 SSD lossless verification at temperature=0 (n=$LIMIT)"
echo "============================================================"
echo

echo "[1/3] PATH A 512 baseline at temp=0 (no SSD)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler argmax \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.0 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/baseline_t0.json" 2>&1 | \
    tee "$RESULTS_DIR/baseline_t0.log"
echo

echo "[2/3] PATH A 512 + SSD k=4 at temp=0"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler argmax \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.0 \
    --top_p 0.95 \
    --speculative_k 4 \
    --out "$RESULTS_DIR/ssd_t0.json" 2>&1 | \
    tee "$RESULTS_DIR/ssd_t0.log"
echo

echo "[3/3] Comparison"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/diff.log"
import json
import os

rd = os.environ["RESULTS_DIR"]
with open(f"{rd}/baseline_t0.json") as f: b = json.load(f)
with open(f"{rd}/ssd_t0.json")      as f: s = json.load(f)

print()
print(f"{'metric':25s}  {'baseline t=0':>15s}  {'SSD k=4 t=0':>15s}  {'delta':>10s}")
print("-" * 75)
print(f"{'pass@1':25s}  {b['pass_at_1_single_shot']:>15.4f}  {s['pass_at_1_single_shot']:>15.4f}  {(s['pass_at_1_single_shot']-b['pass_at_1_single_shot'])*100:>+8.1f}pp")
print(f"{'s/problem':25s}  {b['seconds_per_problem']:>15.2f}  {s['seconds_per_problem']:>15.2f}  {b['seconds_per_problem']/s['seconds_per_problem']:>9.2f}x")
print(f"{'tokens/sec':25s}  {b['tokens_per_second']:>15.1f}  {s['tokens_per_second']:>15.1f}  {s['tokens_per_second']/b['tokens_per_second']:>9.2f}x")
print(f"{'total forwards':25s}  {b['total_forwards']:>15d}  {s['total_forwards']:>15d}  {b['total_forwards']/s['total_forwards']:>9.2f}x")

# Per-problem pass agreement (token-level identity is too strong; pass-fail
# agreement is the right correctness check)
b_pass = {p["task_id"]: p["passed"] for p in b["per_problem"]}
s_pass = {p["task_id"]: p["passed"] for p in s["per_problem"]}
shared = set(b_pass) & set(s_pass)
agree = sum(1 for tid in shared if b_pass[tid] == s_pass[tid])
print()
print(f"per-problem pass/fail agreement: {agree}/{len(shared)} = {agree/len(shared):.2%}")

# Verdict
delta = (s['pass_at_1_single_shot'] - b['pass_at_1_single_shot']) * 100
speedup = b['seconds_per_problem'] / s['seconds_per_problem']
print()
print("=" * 75)
if abs(delta) <= 3.0 and speedup >= 1.5:
    print(f"VERDICT: SSD lossless at temp=0 confirmed.")
    print(f"  pass@1 within {abs(delta):.1f}pp; speedup {speedup:.2f}x.")
    print(f"  Action: ship v0.3.0 with SSD k=4 at temp=0 as recommended speed mode.")
elif abs(delta) <= 3.0:
    print(f"VERDICT: lossless but small speedup ({speedup:.2f}x). Try k=2 or k=8.")
elif speedup >= 1.5:
    print(f"VERDICT: speedup real but {delta:+.1f}pp pass@1 drop on n=20.")
    print(f"  Could be n=20 noise (12.5pp CI). Run n=164 to confirm before shipping.")
else:
    print(f"VERDICT: {speedup:.2f}x speedup, {delta:+.1f}pp drop. Both signals weak.")
    print(f"  SSD implementation may need debugging.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
