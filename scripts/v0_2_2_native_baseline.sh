#!/usr/bin/env bash
# v0.2.2 follow-up: Dream-Coder NATIVE inference baseline on full HE+ (n=164).
#
# Concern: our PATH C/PATH A both score ~0.54 on full HE+. We have NEVER
# measured Dream-Coder's native diffusion_generate() single-shot on full HE+.
# Without that reference we can't tell if mdlm-engine is matching the model's
# true ceiling or leaving 10-20pp on the table.
#
# This script bypasses mdlm-engine entirely and calls the upstream model's
# built-in `diffusion_generate()` API. Same prompts, same sampling settings
# as our PATH C run. If native pass@1 ≈ 0.54, our engine is fine. If native
# pass@1 ≥ 0.65, we have an engine bug.
#
# Cost: ~30-40 min on RTX 5090, ~$0.20.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LIMIT="${LIMIT:-200}"   # 200 ≥ 164 → full HE+
RESULTS_DIR="$WORKSPACE/v0_2_2_native"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.2.2 native baseline — Dream-Coder.diffusion_generate (n=$LIMIT)"
echo "============================================================"
echo

DREAM_PATH="$DREAM_PATH" RESULTS_DIR="$RESULTS_DIR" LIMIT="$LIMIT" \
python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/native.log"
"""Native Dream-Coder inference baseline. No mdlm-engine on the hot path —
we call `model.diffusion_generate()` directly. Results compared against
PATH C and PATH A by the v0.2.2 investigation script.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Reuse the harness's prompt format + check_completion to keep apples-to-apples
from mdlm_engine.bench.harness import (
    _format_prompt, _extract_code, _check_completion,
)

DREAM_PATH = os.environ["DREAM_PATH"]
RESULTS_DIR = Path(os.environ["RESULTS_DIR"])
LIMIT = int(os.environ["LIMIT"])

print(f"Loading {DREAM_PATH} (upstream HF, no mdlm-engine) ...")
tok = AutoTokenizer.from_pretrained(DREAM_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    DREAM_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
).to("cuda").eval()

# Sanity: the model's native diffusion_generate is on the model itself
assert hasattr(model, "diffusion_generate"), \
    "Upstream Dream-Coder should expose diffusion_generate; check trust_remote_code"

print("Loading HumanEval+ ...")
from evalplus.data import get_human_eval_plus
items = list(get_human_eval_plus().items())[:LIMIT]
print(f"  loaded {len(items)} problems")

n_pass = 0
total_tokens = 0
per_problem = []
t_start = time.time()
torch.cuda.reset_peak_memory_stats()

for i, (task_id, row) in enumerate(items):
    t_p_start = time.time()
    prompt_text = _format_prompt(row["prompt"])
    chat = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        return_tensors="pt", return_dict=True, add_generation_prompt=True,
    )
    prompt_ids = chat["input_ids"].to("cuda")
    attn_mask = chat.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to("cuda")

    # Native call. Dream-Coder's diffusion_generate signature (from upstream
    # generation_utils.py) accepts these kwargs. Match our PATH C settings:
    # max_new=256, steps=256, temperature=0.2, top_p=0.95.
    with torch.inference_mode():
        out = model.diffusion_generate(
            inputs=prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=256,
            steps=256,                # Dream-Coder convention: total denoising steps
            temperature=0.2,
            top_p=0.95,
            alg="entropy",            # matches our --sampler entropy
            alg_temp=0.0,
            output_history=False,
            return_dict_in_generate=True,
        )

    # Extract generated portion (after prompt)
    sequences = out.sequences if hasattr(out, "sequences") else out
    new_tokens = sequences[0, prompt_ids.shape[1]:].cpu().tolist()
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    total_tokens += len(new_tokens)

    passed = _check_completion(decoded, row)
    if passed:
        n_pass += 1
    per_problem.append({
        "task_id": task_id,
        "passed": bool(passed),
        "seconds": time.time() - t_p_start,
        "completion_len": len(new_tokens),
    })

    if i < 3:
        print(f"\n--- problem {i} ({task_id}) — passed={passed} ---")
        print(f"completion[:200]: {decoded[:200]!r}")
        print(f"--- end problem {i} ---\n")

    if (i + 1) % 5 == 0:
        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(items)}] pass@1 = {n_pass/(i+1):.4f}  "
              f"avg = {elapsed/(i+1):.2f}s/problem")

wall = time.time() - t_start
result = {
    "config": "native_diffusion_generate",
    "model_path": DREAM_PATH,
    "n_problems_run": len(items),
    "pass_at_1_single_shot": n_pass / max(1, len(items)),
    "seconds_per_problem": wall / max(1, len(items)),
    "tokens_per_second": total_tokens / wall if wall > 0 else 0.0,
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    "wall_seconds": wall,
    "per_problem": per_problem,
}
out_path = RESULTS_DIR / "native.json"
out_path.write_text(json.dumps(result, indent=2))

print()
print("=" * 60)
print("Dream-Coder NATIVE diffusion_generate — full HE+ baseline")
print("=" * 60)
print(f"  problems run:           {result['n_problems_run']}")
print(f"  pass@1 (single-shot):   {result['pass_at_1_single_shot']:.4f}")
print(f"  s/problem:              {result['seconds_per_problem']:.2f}")
print(f"  tokens/sec:             {result['tokens_per_second']:.1f}")
print(f"  peak VRAM (GB):         {result['peak_vram_gb']:.2f}")
print(f"  wall (s):               {result['wall_seconds']:.0f}")
print(f"\nWritten to {out_path}")
PY
echo

echo "============================================================"
echo "3-way comparison: native / PATH C / PATH A"
echo "============================================================"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/3way.log"
import json
import os

inv_dir = "/workspace/v0_2_2_investigation"
nat_dir = os.environ["RESULTS_DIR"]
files = {
    "native":   f"{nat_dir}/native.json",
    "PATH C":   f"{inv_dir}/pathBC.json",
    "PATH A":   f"{inv_dir}/pathA.json",
}
data = {}
for label, path in files.items():
    if os.path.exists(path):
        with open(path) as f:
            data[label] = json.load(f)
    else:
        print(f"  WARN: {path} not found — skipping {label}")

print()
print(f"{'config':12s}  {'pass@1':>8s}  {'s/problem':>10s}  {'tokens/sec':>11s}  {'wall (s)':>9s}")
print("-" * 60)
for label, r in data.items():
    p = r['pass_at_1_single_shot']
    s = r['seconds_per_problem']
    t = r['tokens_per_second']
    w = r['wall_seconds']
    n = r['n_problems_run']
    print(f"{label:12s}  {p:>8.4f}  {s:>10.2f}  {t:>11.1f}  {w:>9.0f}  (n={n})")

# Verdict
if "native" in data and "PATH C" in data:
    delta_engine = data["PATH C"]["pass_at_1_single_shot"] - data["native"]["pass_at_1_single_shot"]
    print(f"\nEngine vs native pass@1: {delta_engine*100:+.1f}pp")
    if abs(delta_engine) <= 0.03:
        print("→ mdlm-engine matches native quality (within 3pp). Engine is not the bottleneck.")
    elif delta_engine < -0.05:
        print(f"→ mdlm-engine is LEAVING {-delta_engine*100:.1f}pp ON THE TABLE vs native.")
        print("  Real engine bug. Investigate scheduler/sampler/template before shipping.")
    elif delta_engine > 0.05:
        print(f"→ mdlm-engine is BETTER than native by {delta_engine*100:.1f}pp (unusual; likely sampling diff).")
    else:
        print(f"→ mdlm-engine within ~5pp of native; bordering noise. Acceptable.")
PY
echo
echo "Artifacts in $RESULTS_DIR"
