#!/usr/bin/env bash
# v0.3.0 acceptance gate — block-level SSD (Redesign A).
#
# Per-step SSD (day-1 design) only got 1.12x speedup because it operated
# on residual masked positions after the regular sampler. Block-level SSD
# (Redesign A) runs on the init forward's logits BEFORE the sampler — gets
# first pick of high-confidence positions. Targets the paper's 2-3x range.
#
# Configs benchmarked at temperature=0, sampler=argmax (lossless regime):
#   1. baseline:   PATH A 512, no SSD
#   2. block-SSD:  PATH A 512 + block_init=True, threshold=0.95
#   3. block-SSD:  + threshold=0.90 (more aggressive)
#   4. block-SSD:  + threshold=0.80 (likely starts losing quality)
#
# Default LIMIT=20 (~10 min, $0.07). LIMIT=200 for full HE+ validation
# (~30 min, $0.20).

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-20}"
RESULTS_DIR="$WORKSPACE/v0_3_0_block_ssd"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

run_config() {
    local label="$1" k="$2" threshold="$3"
    echo "=== $label ==="
    if [[ "$k" == "0" ]]; then
        python3 -m mdlm_engine.bench.harness \
            --adapter dream --model_path "$DREAM_PATH" \
            --use_fastdllm_modeling \
            --cache dkv --scheduler slowfast --sampler argmax \
            --benchmark humaneval_plus --limit "$LIMIT" \
            --max_new_tokens 512 --block_length 32 --steps_per_block 32 \
            --temperature 0.0 --top_p 0.95 \
            --out "$RESULTS_DIR/$label.json" 2>&1 | \
            tee "$RESULTS_DIR/$label.log"
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
            tee "$RESULTS_DIR/$label.log"
    fi
    echo
}

echo "============================================================"
echo "v0.3.0 block-level SSD acceptance gate (n=$LIMIT, temp=0)"
echo "============================================================"
echo

run_config "baseline"   0 "0.0"
run_config "block95"    1 "0.95"
run_config "block90"    1 "0.90"
run_config "block80"    1 "0.80"

echo "============================================================"
echo "Block-level SSD sweep summary"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/summary.log"
import json
import os
from pathlib import Path

rd = Path(os.environ["RESULTS_DIR"])
configs = [
    ("baseline (no SSD)",   "baseline.json"),
    ("block-SSD t=0.95",    "block95.json"),
    ("block-SSD t=0.90",    "block90.json"),
    ("block-SSD t=0.80",    "block80.json"),
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

# Recommendation: max speedup with ≥90% per-problem agreement (n=20 noise floor).
print()
print("Recommendation:")
candidates = []
for label, r in loaded[1:]:
    speedup = base['seconds_per_problem'] / r['seconds_per_problem']
    cur_passes = {p["task_id"]: p["passed"] for p in r["per_problem"]}
    shared = set(base_passes) & set(cur_passes)
    agree_pct = 100 * sum(1 for tid in shared if base_passes[tid] == cur_passes[tid]) / len(shared)
    if agree_pct >= 90:
        candidates.append((speedup, label, agree_pct))

if candidates:
    candidates.sort(reverse=True)
    best_speedup, best_label, best_agree = candidates[0]
    if best_speedup >= 1.50:
        print(f"  → Ship {best_label} as v0.3.0 default ({best_speedup:.2f}x speedup, {best_agree:.0f}% agreement).")
        print(f"     Validate at full HE+ (LIMIT=200) before tagging.")
    elif best_speedup >= 1.20:
        print(f"  → Marginal: {best_label} ({best_speedup:.2f}x, {best_agree:.0f}% agreement). Worth shipping.")
        print(f"     Validate at full HE+ before tagging.")
    else:
        print(f"  → Best: {best_label} only {best_speedup:.2f}x. Block-SSD didn't deliver expected speedup.")
        print(f"     Pivot to Redesign B or ship v0.2.2 final.")
else:
    print("  → No threshold preserves quality (all <90% agreement). Pivot or bail.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
echo
echo "If recommendation is ship-worthy, validate at full HE+:"
echo "  LIMIT=200 bash scripts/v0_3_0_block_ssd.sh"
