#!/usr/bin/env bash
# v0.3.0 step 1: Diverse best-of-N benchmark for Dream-Coder Instruct.
#
# Runs `--diverse 8` on full HumanEval+ — for each problem, tries 8 diverse
# (sampler, scheduler, temp, top_p, steps_per_block) configs in order and
# stops at first pass. Reports both the oracle metric (pass@N = "any of N
# attempts passed") AND the implied single-shot pass@1 from config 0.
#
# Cost estimate at v0.2.2 PATH A 512:
#   - First-config success rate ~0.67 (single-shot pass@1)
#   - avg attempts ~2-3 across 164 problems
#   - ~30-40 s/problem average wall, ~80-110 min total, ~$0.50-0.70.
#
# Output:
#   - $WORKSPACE/v0_3_0_diverse/diverse8.json: per-problem records with
#     n_attempts and winning_config_idx
#   - Headline: pass@N (best-of-8 oracle), pass@1 (config 0 alone)

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-200}"
DIVERSE_N="${DIVERSE_N:-8}"
MAX_NEW="${MAX_NEW:-512}"
RESULTS_DIR="$WORKSPACE/v0_3_0_diverse"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.3.0 step 1: diverse best-of-$DIVERSE_N (Dream PATH A, max_new=$MAX_NEW)"
echo "============================================================"
echo

python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --benchmark humaneval_plus \
    --limit "$LIMIT" \
    --max_new_tokens "$MAX_NEW" \
    --block_length 32 \
    --diverse "$DIVERSE_N" \
    --out "$RESULTS_DIR/diverse${DIVERSE_N}.json" 2>&1 | \
    tee "$RESULTS_DIR/diverse${DIVERSE_N}.log"
echo

# Post-run analysis: which configs are actually firing?
RESULTS_DIR="$RESULTS_DIR" DIVERSE_N="$DIVERSE_N" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/config_usage.log"
import json
import os
from collections import Counter

results_dir = os.environ["RESULTS_DIR"]
n = int(os.environ["DIVERSE_N"])
with open(f"{results_dir}/diverse{n}.json") as f:
    r = json.load(f)

# Distribution of attempts per problem
n_attempts_dist = Counter(p.get("n_attempts", 0) for p in r["per_problem"])
winning_config = Counter(p.get("winning_config_idx", -2) for p in r["per_problem"])

print()
print("Distribution of attempts per problem:")
for k in sorted(n_attempts_dist):
    pct = 100 * n_attempts_dist[k] / len(r["per_problem"])
    print(f"  {k} attempts: {n_attempts_dist[k]:3d} problems ({pct:.1f}%)")

print()
print(f"Winning config index (-1 = failed all {n}):")
for k in sorted(winning_config):
    pct = 100 * winning_config[k] / len(r["per_problem"])
    label = "FAILED" if k == -1 else f"config {k}"
    print(f"  {label:12s}: {winning_config[k]:3d} problems ({pct:.1f}%)")

print()
print("=" * 60)
print(f"pass@1 (config 0 alone):    {r['pass_at_1_single_shot']:.4f}")
print(f"pass@{n} (oracle):              {r['pass_at_1_best_of_n']:.4f}")
print(f"avg attempts per problem:   {sum(p['n_attempts'] for p in r['per_problem'])/len(r['per_problem']):.2f}")
print(f"s/problem (amortized):      {r['seconds_per_problem']:.2f}")
print(f"tokens/sec:                 {r['tokens_per_second']:.1f}")
print(f"total forwards:             {r['total_forwards']}")

# Configs that never won are dead weight; v0.3.1 could prune them
unused = [k for k in range(n) if k not in winning_config or winning_config[k] == 0]
if unused:
    print(f"\nConfigs that never won: {unused}")
    print("  → consider pruning in v0.3.1 to reduce wasted attempts.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
