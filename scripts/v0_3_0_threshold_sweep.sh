#!/usr/bin/env bash
# v0.3.0 SSD confidence-threshold sweep — find the sweet spot.
#
# threshold=0.99 preserved quality (95% agreement, +5pp on n=20 noise) but
# only gave 1.04x speedup because almost no positions clear 99% softmax.
# Lower thresholds = more proposals firing = bigger speedup, BUT once the
# threshold drops too far, commit-order drift kicks in and quality regresses.
#
# This script runs 4 thresholds + baseline in one go (n=20 each, ~12 min):
#   baseline (no SSD)  — control
#   threshold=0.99     — already known: ~1.04x, preserved quality
#   threshold=0.95     — likely sweet spot
#   threshold=0.90     — more aggressive
#   threshold=0.80     — most aggressive (likely starts losing quality)
#
# We pick the threshold with the highest speedup at <3pp pass@1 drop.
# Cost ~$0.10.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_3_0_threshold_sweep"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

run_config() {
    local label="$1" k="$2" threshold="$3"
    echo "=== $label ==="
    if [[ "$k" == "0" ]]; then
        # baseline: no SSD
        python3 -m mdlm_engine.bench.harness \
            --adapter dream --model_path "$DREAM_PATH" \
            --use_fastdllm_modeling \
            --cache dkv --scheduler slowfast --sampler argmax \
            --benchmark humaneval_plus --limit "$LIMIT" \
            --max_new_tokens 512 --block_length 32 --steps_per_block 32 \
            --temperature 0.0 --top_p 0.95 \
            --out "$RESULTS_DIR/$label.json" 2>&1 | \
            tee "$RESULTS_DIR/$label.log" | tail -10
    else
        python3 -m mdlm_engine.bench.harness \
            --adapter dream --model_path "$DREAM_PATH" \
            --use_fastdllm_modeling \
            --cache dkv --scheduler slowfast --sampler argmax \
            --benchmark humaneval_plus --limit "$LIMIT" \
            --max_new_tokens 512 --block_length 32 --steps_per_block 32 \
            --temperature 0.0 --top_p 0.95 \
            --speculative_k "$k" --speculative_threshold "$threshold" \
            --out "$RESULTS_DIR/$label.json" 2>&1 | \
            tee "$RESULTS_DIR/$label.log" | tail -10
    fi
    echo
}

echo "============================================================"
echo "v0.3.0 SSD threshold sweep (n=$LIMIT, temp=0)"
echo "============================================================"
echo

run_config "baseline"  0 "0.0"
run_config "thresh99"  4 "0.99"
run_config "thresh95"  4 "0.95"
run_config "thresh90"  4 "0.90"
run_config "thresh80"  4 "0.80"

echo "============================================================"
echo "Sweep summary"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/summary.log"
import json
import os
from pathlib import Path

rd = Path(os.environ["RESULTS_DIR"])
configs = [
    ("baseline (no SSD)", "baseline.json"),
    ("SSD k=4 t=0.99",    "thresh99.json"),
    ("SSD k=4 t=0.95",    "thresh95.json"),
    ("SSD k=4 t=0.90",    "thresh90.json"),
    ("SSD k=4 t=0.80",    "thresh80.json"),
]
loaded = []
for label, fname in configs:
    p = rd / fname
    if p.exists():
        with open(p) as f:
            loaded.append((label, json.load(f)))
if not loaded:
    print("No results."); raise SystemExit(0)
base_label, base = loaded[0]
base_passes = {p["task_id"]: p["passed"] for p in base["per_problem"]}

print()
print(f"{'config':22s}  {'pass@1':>7s}  {'Δpp':>5s}  {'s/prob':>7s}  {'speedup':>8s}  {'forwards':>9s}  {'agree%':>7s}")
print("-" * 90)
for label, r in loaded:
    p1 = r['pass_at_1_single_shot']
    sp = r['seconds_per_problem']
    fw = r['total_forwards']
    delta = (p1 - base['pass_at_1_single_shot']) * 100
    speedup = base['seconds_per_problem'] / sp if sp > 0 else 0
    if label == base_label:
        agree_str = "—"
    else:
        cur_passes = {p["task_id"]: p["passed"] for p in r["per_problem"]}
        shared = set(base_passes) & set(cur_passes)
        agree = sum(1 for tid in shared if base_passes[tid] == cur_passes[tid])
        agree_str = f"{100*agree/len(shared):.0f}"
    print(f"{label:22s}  {p1:>7.4f}  {delta:>+4.1f}  {sp:>7.2f}  {speedup:>7.2f}x  {fw:>9d}  {agree_str:>6s}")

# Pick recommended threshold: max speedup with |delta| ≤ 3pp.
print()
print("Recommendation:")
candidates = []
for label, r in loaded[1:]:
    delta = (r['pass_at_1_single_shot'] - base['pass_at_1_single_shot']) * 100
    speedup = base['seconds_per_problem'] / r['seconds_per_problem']
    if abs(delta) <= 3.0:
        candidates.append((speedup, label, delta))

if candidates:
    candidates.sort(reverse=True)
    best_speedup, best_label, best_delta = candidates[0]
    if best_speedup >= 1.30:
        print(f"  → Ship {best_label} as v0.3.0 default ({best_speedup:.2f}x speedup, {best_delta:+.1f}pp).")
    elif best_speedup >= 1.10:
        print(f"  → Marginal: {best_label} ({best_speedup:.2f}x, {best_delta:+.1f}pp). Worth shipping if quality holds at full HE+.")
    else:
        print(f"  → All thresholds give <1.10x. Bail to v0.2.2 final; SSD doesn't pay off on this stack.")
else:
    print("  → No threshold preserves quality. Bail to v0.2.2 final.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
