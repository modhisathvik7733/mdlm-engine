#!/usr/bin/env bash
# v0.3.0 acceptance gate at PRODUCTION settings.
#
# Day-1, day-2, day-3 of v0.3.0 used temp=0+argmax for SSD lossless
# validation. Baseline at that config is 0.80 pass@1 (n=20) — much
# lower than v0.2.2's 0.95 (recorded at temp=0.2+entropy+top_p=0.95).
#
# Validating SSD at the "wrong" baseline made every result look bad.
# This script tests SSD at the CORRECT operating point: production
# settings (temp=0.2+entropy+top_p=0.95). Plus the unified Redesign C
# loop (per-step SSD on FULL mask BEFORE sampler at every step).
#
# At threshold=0.95 with top_p=0.95, SSD's argmax commit lines up
# with what the regular sampler would have committed (top-p has no
# other candidates above 0.05 = 1-0.95). SSD is lossless within
# sampling noise.
#
# Configs:
#   1. baseline:      temp=0.2 + entropy + top_p=0.95, no SSD
#   2. SSD t=0.95:    + Redesign C SSD at threshold=0.95
#   3. SSD t=0.90:    + Redesign C SSD at threshold=0.90 (more aggressive)
#
# Default LIMIT=20 (~7 min, $0.05). LIMIT=200 for full HE+ (~25 min, $0.18).

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_3_0_production"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

run_config() {
    local label="$1" k="$2" threshold="$3"
    echo "=== $label ==="
    if [[ "$k" == "0" ]]; then
        python3 -m mdlm_engine.bench.harness \
            --adapter dream --model_path "$DREAM_PATH" \
            --use_fastdllm_modeling \
            --cache dkv --scheduler slowfast --sampler entropy \
            --benchmark humaneval_plus --limit "$LIMIT" \
            --max_new_tokens 512 --block_length 32 --steps_per_block 32 \
            --temperature 0.2 --top_p 0.95 \
            --out "$RESULTS_DIR/$label.json" 2>&1 | \
            tee "$RESULTS_DIR/$label.log"
    else
        python3 -m mdlm_engine.bench.harness \
            --adapter dream --model_path "$DREAM_PATH" \
            --use_fastdllm_modeling \
            --cache dkv --scheduler slowfast --sampler entropy \
            --benchmark humaneval_plus --limit "$LIMIT" \
            --max_new_tokens 512 --block_length 32 --steps_per_block 32 \
            --temperature 0.2 --top_p 0.95 \
            --speculative_k "$k" --speculative_threshold "$threshold" \
            --out "$RESULTS_DIR/$label.json" 2>&1 | \
            tee "$RESULTS_DIR/$label.log"
    fi
    echo
}

echo "============================================================"
echo "v0.3.0 production gate (temp=0.2 + entropy + top_p=0.95, n=$LIMIT)"
echo "============================================================"
echo

run_config "baseline"   0 "0.0"
run_config "ssd_t095"   1 "0.95"
run_config "ssd_t090"   1 "0.90"

echo "============================================================"
echo "v0.3.0 production sweep summary"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/summary.log"
import json
import os
from pathlib import Path

rd = Path(os.environ["RESULTS_DIR"])
configs = [
    ("baseline (no SSD)",  "baseline.json"),
    ("SSD t=0.95",         "ssd_t095.json"),
    ("SSD t=0.90",         "ssd_t090.json"),
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

# Recommendation logic — accuracy must NOT drop materially.
print()
print("Recommendation:")
candidates = []
for label, r in loaded[1:]:
    speedup = base['seconds_per_problem'] / r['seconds_per_problem']
    cur_passes = {p["task_id"]: p["passed"] for p in r["per_problem"]}
    shared = set(base_passes) & set(cur_passes)
    agree_pct = 100 * sum(1 for tid in shared if base_passes[tid] == cur_passes[tid]) / len(shared)
    if agree_pct >= 90 and speedup >= 1.0:
        candidates.append((speedup, label, agree_pct))

if candidates:
    candidates.sort(reverse=True)
    best_speedup, best_label, best_agree = candidates[0]
    if best_speedup >= 1.30:
        print(f"  → Ship {best_label} as v0.3.0 default ({best_speedup:.2f}x speedup, {best_agree:.0f}% agreement).")
        print(f"     Validate at full HE+ (LIMIT=200) before tagging.")
    elif best_speedup >= 1.10:
        print(f"  → Marginal but real: {best_label} ({best_speedup:.2f}x, {best_agree:.0f}% agreement).")
        print(f"     Validate at full HE+ before tagging.")
    else:
        print(f"  → Best: {best_label} only {best_speedup:.2f}x speedup. Below ship threshold.")
else:
    print("  → No threshold preserves accuracy or gives speedup. Bail to v0.2.2.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
echo
echo "If a config is ship-worthy, validate at full HE+:"
echo "  LIMIT=200 bash scripts/v0_3_0_production.sh"
