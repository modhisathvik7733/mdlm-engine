#!/usr/bin/env bash
# v0.2.2 investigation: is PATH A's pass@1 drop (0.95 → 0.85 on 20 problems)
# a real regression or noise?
#
# Approach: run PATH B/C and PATH A on a larger subset (default 50), then
# diff per-problem pass/fail to see which problems regressed. If pass@1
# remains 5-10 pp lower at n=50 with the SAME problems regressing
# repeatedly, the regression is real and worth fixing. If the gap shrinks
# to within ±3 pp or different problems regress on different runs, it's
# noise from a small subset.
#
# Cost: ~50 problems × (8.4 + 11.3) = ~16 min wall ≈ $0.10 of vast.ai time.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-50}"
RESULTS_DIR="$WORKSPACE/v0_2_2_investigation"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.2.2 investigation — pass@1 regression on PATH A (limit=$LIMIT)"
echo "============================================================"
echo

echo "[1/3] PATH B/C baseline (upstream HF Dream, no caching) — limit=$LIMIT"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens 256 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/pathBC.json" 2>&1 | \
    tee "$RESULTS_DIR/pathBC.log"
echo

echo "[2/3] PATH A (fast_dllm-patched modeling, dual_cache) — limit=$LIMIT"
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
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/pathA.json" 2>&1 | \
    tee "$RESULTS_DIR/pathA.log"
echo

echo "[3/3] Per-problem diff"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/diff.log"
import json
import os

results_dir = os.environ["RESULTS_DIR"]
with open(f"{results_dir}/pathBC.json") as f: bc = json.load(f)
with open(f"{results_dir}/pathA.json")  as f: a  = json.load(f)

bc_by_id = {p["task_id"]: p for p in bc["per_problem"]}
a_by_id  = {p["task_id"]: p for p in a["per_problem"]}
all_ids = sorted(set(bc_by_id) & set(a_by_id))

regressed = [tid for tid in all_ids if bc_by_id[tid]["passed"] and not a_by_id[tid]["passed"]]
gained    = [tid for tid in all_ids if not bc_by_id[tid]["passed"] and a_by_id[tid]["passed"]]
both_pass = [tid for tid in all_ids if bc_by_id[tid]["passed"] and a_by_id[tid]["passed"]]
both_fail = [tid for tid in all_ids if not bc_by_id[tid]["passed"] and not a_by_id[tid]["passed"]]

print()
print(f"{'metric':25s}  {'PATH B/C':>12s}  {'PATH A':>12s}  {'delta':>10s}")
print("-" * 65)
print(f"{'n problems':25s}  {len(bc_by_id):>12d}  {len(a_by_id):>12d}")
print(f"{'pass@1':25s}  {bc['pass_at_1_single_shot']:>12.4f}  {a['pass_at_1_single_shot']:>12.4f}  {(a['pass_at_1_single_shot']-bc['pass_at_1_single_shot'])*100:>+9.1f}pp")
print(f"{'s/problem':25s}  {bc['seconds_per_problem']:>12.2f}  {a['seconds_per_problem']:>12.2f}  {bc['seconds_per_problem']/a['seconds_per_problem']:>9.2f}x")
print(f"{'tokens/sec':25s}  {bc['tokens_per_second']:>12.1f}  {a['tokens_per_second']:>12.1f}  {a['tokens_per_second']/bc['tokens_per_second']:>9.2f}x")
print(f"{'total forwards':25s}  {bc['total_forwards']:>12d}  {a['total_forwards']:>12d}")
print()
print(f"both pass:    {len(both_pass):3d}  ({100*len(both_pass)/len(all_ids):.0f}%)")
print(f"both fail:    {len(both_fail):3d}  ({100*len(both_fail)/len(all_ids):.0f}%)")
print(f"gained on A:  {len(gained):3d}  ({100*len(gained)/len(all_ids):.0f}%)  PATH A passed but B/C failed")
print(f"REGRESSED:    {len(regressed):3d}  ({100*len(regressed)/len(all_ids):.0f}%)  PATH B/C passed but A failed")
print()
if regressed:
    print("Regressed task ids:")
    for tid in regressed:
        a_sec = a_by_id[tid]["seconds"]
        bc_sec = bc_by_id[tid]["seconds"]
        a_len = a_by_id[tid]["completion_len"]
        bc_len = bc_by_id[tid]["completion_len"]
        print(f"  {tid:20s}  PATH A: {a_sec:5.1f}s len={a_len:3d}  |  PATH B/C: {bc_sec:5.1f}s len={bc_len:3d}")
if gained:
    print()
    print("Gained task ids (PATH A passed, B/C failed):")
    for tid in gained:
        print(f"  {tid}")

# Verdict
net = len(gained) - len(regressed)
print()
print("=" * 65)
if abs(net) <= 1:
    print(f"VERDICT: noise — net delta is {net:+d} problems on {len(all_ids)} runs.")
    print("v0.2.1 pass@1 0.85 vs v0.2.0 0.95 was likely 20-problem-subset variance.")
elif net <= -3:
    print(f"VERDICT: real regression — net delta is {net:+d} problems on {len(all_ids)} runs.")
    print("Path A consistently loses on the SAME problems. Investigate K/V drift")
    print("at masked positions reused across cached steps.")
else:
    print(f"VERDICT: small regression — net delta is {net:+d} on {len(all_ids)} runs.")
    print("Borderline. Consider running larger n or accepting as cost of caching.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
