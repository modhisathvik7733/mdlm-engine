#!/usr/bin/env bash
# v0.2.2 follow-up: PATH A at higher generation budget.
#
# The user's notes (Dream-Coder scripts README): "Default paper config:
# steps=768, max_new_tokens=768 — we use 128/256 for speed without
# accuracy loss" — but that "without accuracy loss" was measured on
# best-of-8 oracle with n=20. SINGLE-SHOT might lose accuracy at the
# lower budget.
#
# 7 of the regressed problems on PATH A's full HE+ run had completion_len
# = 256 (hit the cap) — likely truncated mid-function. This script runs
# PATH A with max_new=512 (2× the budget) to see if pass@1 recovers.
#
# Default: max_new=512, block_length=32 → 16 blocks (vs 8 at max_new=256).
# steps_per_block stays at 32 (so 16 × 32 = 512 total denoising steps).
#
# Cost: ~44 min on RTX 5090 (~2× the 256-budget run), ~$0.27.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-200}"
MAX_NEW="${MAX_NEW:-512}"
SRC_DIR="$WORKSPACE/v0_2_2_investigation"
OUT_DIR="$WORKSPACE/v0_2_2_budget$MAX_NEW"
mkdir -p "$OUT_DIR"

cd "$REPO_DIR"

if [[ ! -f "$SRC_DIR/pathA.json" ]]; then
    echo "ERROR: expected $SRC_DIR/pathA.json from prior run."
    echo "Run scripts/v0_2_2_investigation.sh first to establish max_new=256 baseline."
    exit 1
fi

echo "============================================================"
echo "v0.2.2 follow-up — PATH A at max_new=$MAX_NEW (vs 256 baseline)"
echo "============================================================"
echo

echo "[1/2] PATH A with max_new_tokens=$MAX_NEW (limit=$LIMIT)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW" \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$OUT_DIR/pathA.json" 2>&1 | \
    tee "$OUT_DIR/pathA.log"
echo

echo "[2/2] Per-problem diff: PATH A 256 vs PATH A $MAX_NEW"
SRC_DIR="$SRC_DIR" OUT_DIR="$OUT_DIR" MAX_NEW="$MAX_NEW" python3 - <<'PY' 2>&1 | tee "$OUT_DIR/diff.log"
import json
import os

src = os.environ["SRC_DIR"]
out = os.environ["OUT_DIR"]
mn = os.environ["MAX_NEW"]

with open(f"{src}/pathA.json")  as f: a256 = json.load(f)
with open(f"{out}/pathA.json")  as f: aN   = json.load(f)
with open(f"{src}/pathBC.json") as f: bc   = json.load(f)

a256_by_id = {p["task_id"]: p for p in a256["per_problem"]}
aN_by_id   = {p["task_id"]: p for p in aN["per_problem"]}
bc_by_id   = {p["task_id"]: p for p in bc["per_problem"]}
all_ids = sorted(set(a256_by_id) & set(aN_by_id) & set(bc_by_id))

print()
print(f"{'metric':25s}  {'PATH C 256':>12s}  {'PATH A 256':>12s}  {f'PATH A {mn}':>12s}")
print("-" * 72)
print(f"{'pass@1':25s}  {bc['pass_at_1_single_shot']:>12.4f}  {a256['pass_at_1_single_shot']:>12.4f}  {aN['pass_at_1_single_shot']:>12.4f}")
print(f"{'s/problem':25s}  {bc['seconds_per_problem']:>12.2f}  {a256['seconds_per_problem']:>12.2f}  {aN['seconds_per_problem']:>12.2f}")
print(f"{'tokens/sec':25s}  {bc['tokens_per_second']:>12.1f}  {a256['tokens_per_second']:>12.1f}  {aN['tokens_per_second']:>12.1f}")
print(f"{'total forwards':25s}  {bc['total_forwards']:>12d}  {a256['total_forwards']:>12d}  {aN['total_forwards']:>12d}")
print(f"{'wall (s)':25s}  {bc['wall_seconds']:>12.0f}  {a256['wall_seconds']:>12.0f}  {aN['wall_seconds']:>12.0f}")
print()

# Higher-budget recovery analysis: did problems that hit max_len=256 on the
# old run pass at the new budget?
truncated_at_256 = [tid for tid in all_ids
                    if a256_by_id[tid]["completion_len"] == 256
                    and not a256_by_id[tid]["passed"]]
recovered = [tid for tid in truncated_at_256 if aN_by_id[tid]["passed"]]
still_fail = [tid for tid in truncated_at_256 if not aN_by_id[tid]["passed"]]
print(f"Of {len(truncated_at_256)} problems that PATH A 256 truncated AND failed:")
print(f"  → recovered with PATH A {mn}: {len(recovered)}")
print(f"  → still failing on PATH A {mn}: {len(still_fail)}")
if recovered:
    print(f"  recovered examples: {recovered[:8]}")

# Net pass@1 change
gained = [tid for tid in all_ids if not a256_by_id[tid]["passed"] and aN_by_id[tid]["passed"]]
lost   = [tid for tid in all_ids if a256_by_id[tid]["passed"] and not aN_by_id[tid]["passed"]]
print(f"\nGained at PATH A {mn}: {len(gained)} problems")
print(f"Lost at PATH A {mn}:   {len(lost)} problems")
print(f"Net delta: {len(gained) - len(lost):+d} problems")

# Verdict
delta_vs_256 = (aN['pass_at_1_single_shot'] - a256['pass_at_1_single_shot']) * 100
delta_vs_bc  = (aN['pass_at_1_single_shot'] - bc['pass_at_1_single_shot']) * 100
cost_ratio   = aN['seconds_per_problem'] / a256['seconds_per_problem']
print()
print("=" * 72)
print(f"PATH A {mn} vs PATH A 256: pass@1 {delta_vs_256:+.1f}pp, cost {cost_ratio:.2f}x")
print(f"PATH A {mn} vs PATH C 256: pass@1 {delta_vs_bc:+.1f}pp")
if delta_vs_256 >= 5:
    print(f"VERDICT: budget mattered. Higher max_new recovers {delta_vs_256:+.1f}pp.")
    print(f"Action: consider max_new={mn} as recommended preset for quality.")
elif delta_vs_256 >= 2:
    print(f"VERDICT: budget helped a little ({delta_vs_256:+.1f}pp).")
    print(f"Tradeoff: {cost_ratio:.2f}x slower for {delta_vs_256:+.1f}pp gain. Marginal.")
else:
    print(f"VERDICT: budget didn't help meaningfully ({delta_vs_256:+.1f}pp).")
    print(f"Pass@1 ceiling at this scheduler/sampler is ~{a256['pass_at_1_single_shot']:.2f}.")
    print(f"To go higher, try diverse-best-of-N or different sampler/scheduler.")
PY
echo
echo "Artifacts in $OUT_DIR"
