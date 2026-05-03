#!/usr/bin/env bash
# v0.3.0 acceptance gate — DIVERSE-8 + SSD (the recipe that gets you 0.95+).
#
# v0.3.0's single-shot SSD gave 2.34x speedup at 0.80 pass@1 on n=20.
# But the user's previous Dream-Coder runs (per diffucoder_experiments
# README) hit 100% (20/20) on n=20 with diverse best-of-8. Single-shot
# is the wrong metric for the user's workflow — they want pass@N oracle.
#
# This script combines:
#   - --diverse 8         (the 8-config best-of-N from harness.py)
#   - --speculative_k 1   (SSD on each attempt)
#   - --speculative_threshold 0.95  (lossless within sampling noise)
#
# Each of the 8 attempts uses SSD internally (~2.34x faster per attempt).
# First-pass-acceptance: stops at first config that passes. Average
# attempts ~2 → amortized ~8 s/problem at ~0.95+ pass@8.
#
# Configs benchmarked (n=20, ~15-20 min, ~$0.10-0.15):
#   1. baseline:     diverse-8, NO SSD
#   2. diverse+SSD:  diverse-8 + SSD k=1, threshold=0.95
#   3. diverse+SSD:  diverse-8 + SSD k=1, threshold=0.90 (more aggressive)

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_3_0_diverse_ssd"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

run_diverse_config() {
    local label="$1" ssd_k="$2" threshold="$3"
    echo "=== $label ==="
    python3 -m mdlm_engine.bench.harness \
        --adapter dream --model_path "$DREAM_PATH" \
        --use_fastdllm_modeling \
        --cache dkv \
        --benchmark humaneval_plus --limit "$LIMIT" \
        --max_new_tokens 512 --block_length 32 \
        --diverse 8 \
        --speculative_k "$ssd_k" \
        --speculative_threshold "$threshold" \
        --out "$RESULTS_DIR/$label.json" 2>&1 | \
        tee "$RESULTS_DIR/$label.log"
    echo
}

echo "============================================================"
echo "v0.3.0 diverse-8 + SSD gate (n=$LIMIT)"
echo "============================================================"
echo

run_diverse_config "div8_no_ssd"     0 "0.0"
run_diverse_config "div8_ssd_t095"   1 "0.95"
run_diverse_config "div8_ssd_t090"   1 "0.90"

echo "============================================================"
echo "Diverse-8 + SSD summary"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/summary.log"
import json
import os
from pathlib import Path

rd = Path(os.environ["RESULTS_DIR"])
configs = [
    ("diverse-8 (no SSD)",     "div8_no_ssd.json"),
    ("diverse-8 + SSD t=0.95", "div8_ssd_t095.json"),
    ("diverse-8 + SSD t=0.90", "div8_ssd_t090.json"),
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

print()
print(f"{'config':28s}  {'pass@N':>7s}  {'pass@1':>7s}  {'s/prob':>7s}  {'speedup':>8s}  {'avg_att':>7s}  {'forwards':>9s}")
print("-" * 95)
for label, r in loaded:
    pN = r['pass_at_1_best_of_n']
    p1 = r.get('pass_at_1_single_shot', 0.0)
    sp = r['seconds_per_problem']
    fw = r['total_forwards']
    avg_att = sum(p.get('n_attempts', 1) for p in r.get('per_problem', [])) / max(1, len(r.get('per_problem', [1])))
    speedup = base['seconds_per_problem'] / sp if sp > 0 else 0
    print(f"{label:28s}  {pN:>7.4f}  {p1:>7.4f}  {sp:>7.2f}  {speedup:>7.2f}x  {avg_att:>7.2f}  {fw:>9d}")

# Recommendation: max speedup with pass@N within 5pp of baseline.
print()
print("Recommendation:")
candidates = []
for label, r in loaded[1:]:
    speedup = base['seconds_per_problem'] / r['seconds_per_problem']
    delta = (r['pass_at_1_best_of_n'] - base['pass_at_1_best_of_n']) * 100
    if abs(delta) <= 5:
        candidates.append((speedup, label, delta, r['pass_at_1_best_of_n']))

if candidates:
    candidates.sort(reverse=True)
    best_speedup, best_label, best_delta, best_passN = candidates[0]
    print(f"  → Ship {best_label} as v0.3.0 default")
    print(f"     pass@8 = {best_passN:.4f} (Δ {best_delta:+.1f}pp), {best_speedup:.2f}x faster")
    print(f"     Validate at full HE+ (LIMIT=200) before tagging.")
else:
    print("  → All thresholds drop pass@N by >5pp. Lower SSD threshold or bail.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
echo
echo "If green, validate at full HE+:"
echo "  LIMIT=200 bash scripts/v0_3_0_diverse_ssd.sh"
