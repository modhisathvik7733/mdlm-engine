#!/usr/bin/env bash
# v0.2.2 follow-up: does halving steps_per_block recover pass@1 on PATH A?
#
# Hypothesis: PATH A's pass@1 regression vs PATH C is from K/V drift at
# future-masked positions across iter steps within a block. With
# steps_per_block=32, those future K/V are reused 31 times. With
# steps_per_block=16, only 15 times — half the drift exposure.
#
# Re-uses the previous PATH C and PATH A (steps=32) results from
# /workspace/v0_2_2_investigation/. Adds one new run: PATH A with
# steps_per_block=16. Per-problem diff against the existing PATH A 32-step
# results to see WHICH problems recover.
#
# Cost: ~7 min on RTX 5090, ~$0.05.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-50}"
SRC_DIR="$WORKSPACE/v0_2_2_investigation"
OUT_DIR="$WORKSPACE/v0_2_2_steps16"
mkdir -p "$OUT_DIR"

cd "$REPO_DIR"

if [[ ! -f "$SRC_DIR/pathA.json" || ! -f "$SRC_DIR/pathBC.json" ]]; then
    echo "ERROR: expected $SRC_DIR/pathA.json and pathBC.json from prior run."
    echo "Run scripts/v0_2_2_investigation.sh first."
    exit 1
fi

echo "============================================================"
echo "v0.2.2 follow-up — PATH A steps_per_block=16 (vs 32 baseline)"
echo "============================================================"
echo

echo "[1/2] PATH A with steps_per_block=16 (limit=$LIMIT)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 256 \
    --block_length 32 \
    --steps_per_block 16 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$OUT_DIR/pathA_steps16.json" 2>&1 | \
    tee "$OUT_DIR/pathA_steps16.log" | tail -8
echo

echo "[2/2] Per-problem diff: PATH C 32 / PATH A 32 / PATH A 16"
SRC_DIR="$SRC_DIR" OUT_DIR="$OUT_DIR" python3 - <<'PY' 2>&1 | tee "$OUT_DIR/diff.log"
import json
import os

src = os.environ["SRC_DIR"]
out = os.environ["OUT_DIR"]
with open(f"{src}/pathBC.json")        as f: bc = json.load(f)
with open(f"{src}/pathA.json")         as f: a32 = json.load(f)
with open(f"{out}/pathA_steps16.json") as f: a16 = json.load(f)

bc_by_id  = {p["task_id"]: p for p in bc["per_problem"]}
a32_by_id = {p["task_id"]: p for p in a32["per_problem"]}
a16_by_id = {p["task_id"]: p for p in a16["per_problem"]}
all_ids = sorted(set(bc_by_id) & set(a32_by_id) & set(a16_by_id))

print()
print(f"{'metric':25s}  {'PATH C 32':>12s}  {'PATH A 32':>12s}  {'PATH A 16':>12s}")
print("-" * 72)
print(f"{'pass@1':25s}  {bc['pass_at_1_single_shot']:>12.4f}  {a32['pass_at_1_single_shot']:>12.4f}  {a16['pass_at_1_single_shot']:>12.4f}")
print(f"{'s/problem':25s}  {bc['seconds_per_problem']:>12.2f}  {a32['seconds_per_problem']:>12.2f}  {a16['seconds_per_problem']:>12.2f}")
print(f"{'tokens/sec':25s}  {bc['tokens_per_second']:>12.1f}  {a32['tokens_per_second']:>12.1f}  {a16['tokens_per_second']:>12.1f}")
print(f"{'total forwards':25s}  {bc['total_forwards']:>12d}  {a32['total_forwards']:>12d}  {a16['total_forwards']:>12d}")
print()

# Regression analysis: which problems did PATH A 16 recover vs PATH A 32?
prev_regressed = [tid for tid in all_ids if bc_by_id[tid]["passed"] and not a32_by_id[tid]["passed"]]
prev_gained    = [tid for tid in all_ids if not bc_by_id[tid]["passed"] and a32_by_id[tid]["passed"]]

a16_now_passes = [tid for tid in prev_regressed if a16_by_id[tid]["passed"]]
a16_still_fails = [tid for tid in prev_regressed if not a16_by_id[tid]["passed"]]
a16_lost_gain = [tid for tid in prev_gained if not a16_by_id[tid]["passed"]]

print(f"Of {len(prev_regressed)} problems PATH A 32 regressed on:")
print(f"  → recovered with PATH A 16: {len(a16_now_passes)}")
print(f"  → still failing on PATH A 16: {len(a16_still_fails)}")
if a16_now_passes:
    print(f"  recovered: {', '.join(a16_now_passes)}")
if a16_still_fails:
    print(f"  still failing: {', '.join(a16_still_fails)}")
print()
print(f"Of {len(prev_gained)} problems PATH A 32 gained over PATH C:")
print(f"  → still pass on PATH A 16: {len(prev_gained) - len(a16_lost_gain)}")
print(f"  → lost on PATH A 16: {len(a16_lost_gain)}")
if a16_lost_gain:
    print(f"  lost: {', '.join(a16_lost_gain)}")

# New regressed problems on PATH A 16 (that PATH A 32 didn't lose on)
new_regressed_16 = [tid for tid in all_ids
                    if bc_by_id[tid]["passed"]
                    and a32_by_id[tid]["passed"]
                    and not a16_by_id[tid]["passed"]]
if new_regressed_16:
    print(f"\nNEW regressions on PATH A 16 (passed under PATH A 32): {new_regressed_16}")

# Verdict
delta_a16_vs_bc = (a16['pass_at_1_single_shot'] - bc['pass_at_1_single_shot']) * 100
delta_a16_vs_a32 = (a16['pass_at_1_single_shot'] - a32['pass_at_1_single_shot']) * 100
print()
print("=" * 72)
if delta_a16_vs_a32 >= 4 and abs(delta_a16_vs_bc) <= 3:
    print(f"VERDICT: drift confirmed.")
    print(f"  PATH A 16 vs PATH C 32: {delta_a16_vs_bc:+.1f}pp (within noise)")
    print(f"  PATH A 16 vs PATH A 32: {delta_a16_vs_a32:+.1f}pp (recovered)")
    print(f"Action: make steps_per_block=16 the default for PATH A.")
elif delta_a16_vs_a32 >= 2:
    print(f"VERDICT: drift partially confirmed.")
    print(f"  PATH A 16 vs PATH A 32: {delta_a16_vs_a32:+.1f}pp (some recovery)")
    print(f"  PATH A 16 vs PATH C 32: {delta_a16_vs_bc:+.1f}pp")
    print(f"Action: investigate further (mid-block re-init?) or accept trade-off.")
else:
    print(f"VERDICT: drift NOT the cause (or fix doesn't help).")
    print(f"  PATH A 16 vs PATH A 32: {delta_a16_vs_a32:+.1f}pp")
    print(f"Action: investigate force-commit cost or scheduler interaction.")
PY
echo
echo "Artifacts in $OUT_DIR"
