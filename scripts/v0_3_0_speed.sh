#!/usr/bin/env bash
# v0.3.0 speed gate. Runs the 4 speed-focused configurations on
# Dream-Coder HumanEval+ subset (default n=20) and prints a comparison.
#
# Configs benchmarked:
#   1. v0.2.2 baseline (PATH A 512, no MXFP8, no compile, no SSD)
#   2. v0.3.0 candidate: PATH A 512 + SSD k=4
#   3. v0.3.0 + MXFP8 (gated on day-1 spike PASS)
#   4. v0.3.0 + MXFP8 + compile (full stack)
#
# Default n=20: speed signal is reliable for RELATIVE comparison even on
# small subset. Quality was already validated at full HE+ in v0.2.2's
# acceptance run (BENCHMARKS.md: 0.6707 pass@1). Override LIMIT=200 if
# you want absolute pass@1 numbers from this run.
#
# Cost (n=20): ~12-15 min wall, ~$0.10 of vast.ai 5090 time.
# Cost (n=164): ~50-90 min wall, ~$0.50.
#
# Override via env: SKIP_MXFP8=1 (if spike said FAIL) or SKIP_COMPILE=1.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
SKIP_MXFP8="${SKIP_MXFP8:-0}"
SKIP_COMPILE="${SKIP_COMPILE:-0}"
SPECULATIVE_K="${SPECULATIVE_K:-4}"
RESULTS_DIR="$WORKSPACE/v0_3_0_speed"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.3.0 acceptance gate — Dream-Coder PATH A 512 + speed levers"
echo "============================================================"
echo "  LIMIT=$LIMIT  SPECULATIVE_K=$SPECULATIVE_K"
echo "  SKIP_MXFP8=$SKIP_MXFP8  SKIP_COMPILE=$SKIP_COMPILE"
echo

# ---- Config 1: v0.2.2 baseline (PATH A 512, no SSD, no quant, no compile) ----
echo "[1/4] v0.2.2 baseline (PATH A 512, single-shot, no SSD)"
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
    --out "$RESULTS_DIR/baseline.json" 2>&1 | \
    tee "$RESULTS_DIR/baseline.log"
echo

# ---- Config 2: + SSD ----
echo "[2/4] v0.3.0 candidate: PATH A 512 + SSD k=$SPECULATIVE_K"
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
    --speculative_k "$SPECULATIVE_K" \
    --out "$RESULTS_DIR/ssd.json" 2>&1 | \
    tee "$RESULTS_DIR/ssd.log"
echo

# ---- Config 3: + SSD + MXFP8 (skip if spike failed) ----
if [[ "$SKIP_MXFP8" != "1" ]]; then
    echo "[3/4] v0.3.0: PATH A 512 + SSD + MXFP8"
    python3 -m mdlm_engine.bench.harness \
        --adapter dream \
        --model_path "$DREAM_PATH" \
        --use_fastdllm_modeling \
        --quant mxfp8 \
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
        --speculative_k "$SPECULATIVE_K" \
        --out "$RESULTS_DIR/ssd_mxfp8.json" 2>&1 | \
        tee "$RESULTS_DIR/ssd_mxfp8.log"
    echo
else
    echo "[3/4] SKIPPED — SKIP_MXFP8=1 (spike said FAIL)"
    echo
fi

# ---- Config 4: + SSD + MXFP8 + compile (skip if either bust) ----
if [[ "$SKIP_MXFP8" != "1" && "$SKIP_COMPILE" != "1" ]]; then
    echo "[4/4] v0.3.0 stretch: PATH A 512 + SSD + MXFP8 + compile"
    python3 -m mdlm_engine.bench.harness \
        --adapter dream \
        --model_path "$DREAM_PATH" \
        --use_fastdllm_modeling \
        --quant mxfp8 \
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
        --speculative_k "$SPECULATIVE_K" \
        --out "$RESULTS_DIR/full_stack.json" 2>&1 | \
        tee "$RESULTS_DIR/full_stack.log"
    echo
else
    echo "[4/4] SKIPPED — MXFP8 or compile disabled"
    echo
fi

# ---- Speedup table ----
echo "============================================================"
echo "v0.3.0 speedup summary"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/summary.log"
import json
import os
from pathlib import Path

rd = Path(os.environ["RESULTS_DIR"])
configs = [
    ("baseline (v0.2.2)", "baseline.json"),
    ("+ SSD", "ssd.json"),
    ("+ SSD + MXFP8", "ssd_mxfp8.json"),
    ("+ SSD + MXFP8 + compile", "full_stack.json"),
]
loaded = []
for label, fname in configs:
    p = rd / fname
    if p.exists():
        with open(p) as f:
            loaded.append((label, json.load(f)))

if not loaded:
    print("No results to summarize.")
    raise SystemExit(0)

base_label, base = loaded[0]
print(f"{'config':32s}  {'pass@1':>8s}  {'s/prob':>8s}  {'tok/sec':>9s}  {'forwards':>10s}  {'speedup':>9s}")
print("-" * 90)
for label, r in loaded:
    p1 = r['pass_at_1_single_shot']
    sp = r['seconds_per_problem']
    ts = r['tokens_per_second']
    fw = r['total_forwards']
    speedup = base['seconds_per_problem'] / sp if sp > 0 else 0
    print(f"{label:32s}  {p1:>8.4f}  {sp:>8.2f}  {ts:>9.1f}  {fw:>10d}  {speedup:>8.2f}x")

# Quality gate: pass@1 must stay within 3pp of baseline.
print()
for label, r in loaded[1:]:
    delta = (r['pass_at_1_single_shot'] - base['pass_at_1_single_shot']) * 100
    sym = "✓" if abs(delta) <= 3.0 else "✗"
    print(f"  {sym} {label}: pass@1 delta = {delta:+.1f}pp")
PY
echo
echo "Artifacts in $RESULTS_DIR"
