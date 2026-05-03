#!/usr/bin/env bash
# Phase 2.1 (v0.2.1) acceptance gate — engages PATH A via bundled fast_dllm modeling.
#
# Differences from phase2_acceptance.sh:
#   1. Calls verify_dual_cache.py with --use_fastdllm_modeling: should print PATH A
#   2. Bench Dream with --use_fastdllm_modeling: target s/problem ≤ 6.0 (~2x v0.2.0)
#   3. Cache equivalence: drops 'none' vs others as a gate; instead asserts that
#      block ≡ dkv at temp 0 (PATH A actually engages caching, so 'none' will
#      legitimately diverge if the cache is doing its job).
#   4. Side-by-side bench: also runs Dream WITHOUT --use_fastdllm_modeling for
#      direct A/B speed comparison on the same hardware.
#   5. NaN check uses --use_fastdllm_modeling.
#
# Run on a vast.ai box AFTER bootstrap_vastai.sh + git pull. ~30-45 min.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LLADA_PATH="${LLADA_PATH:-GSAI-ML/LLaDA-8B-Base}"
RESULTS_DIR="$WORKSPACE/phase2_1_acceptance"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "Phase 2.1 acceptance gate — mdlm-engine v0.2.1 (PATH A engaged)"
echo "============================================================"
echo

# ---- 0. Show env ----
echo "[env]"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
python3 -c "import torch; print(f'  torch {torch.__version__} CUDA {torch.version.cuda}')"
echo

# ---- 1. Test suite ----
echo "[1/8] Test suite (pytest tests/)"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/01_pytest.log" | tail -6
echo

# ---- 2. PATH A verification: load fast_dllm-patched, inspect signature ----
echo "[2/8] PATH A verification — fast_dllm-patched modeling exposes dual_cache + replace_position"
python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/02_path_a_verify.log"
import inspect
import torch
from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm

print("Loading Dream-Coder with fast_dllm-patched modeling ...")
model = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cpu").eval()
sig = inspect.signature(model.forward)
params = list(sig.parameters)
print("  forward params:", params)
print("  dual_cache:       ", "dual_cache" in params)
print("  replace_position: ", "replace_position" in params)
print("  past_key_values:  ", "past_key_values" in params)
print("  use_cache:        ", "use_cache" in params)
print()
if "dual_cache" in params and "replace_position" in params:
    print("VERDICT: PATH A engaged. DreamAdapter will use dual_cache=True.")
else:
    print("VERDICT: PATH A NOT engaged — fast_dllm modeling overlay failed.")
    raise SystemExit(1)
PY
echo

# ---- 3. Dream benchmark on PATH B (baseline; should match v0.2.0) ----
echo "[3/8] Dream baseline (PATH B → C; v0.2.0 behavior)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/03_dream_pathB.json" 2>&1 | \
    tee "$RESULTS_DIR/03_dream_pathB.log"
echo

# ---- 4. Dream benchmark on PATH A (THE speedup test) ----
echo "[4/8] Dream with PATH A (--use_fastdllm_modeling): the speedup gate"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --use_fastdllm_modeling \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/04_dream_pathA.json" 2>&1 | \
    tee "$RESULTS_DIR/04_dream_pathA.log"
echo

# ---- 5. Cache equivalence on PATH A: block ≡ dkv (none diverges legitimately now) ----
echo "[5/8] Cache equivalence on PATH A: block vs dkv must match at temp 0"
python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/05_cache_equivalence_pathA.log"
import torch
from transformers import AutoTokenizer
from mdlm_engine import DiffusionEngine
from mdlm_engine.adapters.dream import DreamAdapter
from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm

PROMPTS = [
    "Write add(a, b).",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write merge_sort(arr) and return the sorted list.",
]

tok = AutoTokenizer.from_pretrained("Dream-org/Dream-Coder-v0-Instruct-7B", trust_remote_code=True)
model = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cuda").eval()
adapter = DreamAdapter(model=model, tokenizer=tok)
print(f"  adapter._caps.path = {adapter._caps.path} (must be 'A')")
assert adapter._caps.path == "A", "PATH A not engaged — fast_dllm overlay failed"

all_pass = True
for prompt in PROMPTS:
    print(f"\n[prompt] {prompt}")
    seqs = {}
    for kind in ("block", "dkv"):
        engine = DiffusionEngine(model, adapter=adapter, cache=kind, sampler="argmax", scheduler="slowfast")
        ids = adapter.apply_chat_template([{"role": "user", "content": prompt}]).to("cuda")
        out = engine.generate(ids, max_new_tokens=64, block_length=32, steps_per_block=16, temperature=0.0)
        seqs[kind] = out.sequences.cpu()
        print(f"  cache={kind}: shape={tuple(seqs[kind].shape)}")
    L = min(seqs['block'].shape[1], seqs['dkv'].shape[1])
    diff = int((seqs['block'][:, :L] != seqs['dkv'][:, :L]).sum())
    if diff:
        print(f"  DRIFT: {diff} differing tokens")
        all_pass = False
    else:
        print(f"  OK: identical on common prefix ({L} tokens)")

print(f"\nVERDICT: {'PASS' if all_pass else 'FAIL'}")
raise SystemExit(0 if all_pass else 1)
PY
echo

# ---- 6. NaN-freedom on PATH A ----
echo "[6/8] NaN-freedom (100 generations on PATH A)"
python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/06_nan_pathA.log"
import torch
from transformers import AutoTokenizer
from mdlm_engine import DiffusionEngine
from mdlm_engine.adapters.dream import DreamAdapter
from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm

tok = AutoTokenizer.from_pretrained("Dream-org/Dream-Coder-v0-Instruct-7B", trust_remote_code=True)
model = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cuda").eval()
adapter = DreamAdapter(model=model, tokenizer=tok)
print(f"  PATH: {adapter._caps.path} (expect A)")
engine = DiffusionEngine(model, adapter=adapter, cache="dkv", sampler="argmax", scheduler="slowfast")
prompt = adapter.apply_chat_template([{"role": "user", "content": "Write add(a, b)."}]).to("cuda")
nan_count = 0
for i in range(100):
    out = engine.generate(prompt, max_new_tokens=64, block_length=32, steps_per_block=16, temperature=0.0)
    if torch.isnan(out.sequences.float()).any():
        nan_count += 1
print(f"  100 generations done. NaN count: {nan_count} (must be 0 to pass)")
PY
echo

# ---- 7. LLaDA portability smoke (unchanged) ----
echo "[7/8] LLaDA portability smoke (unchanged from v0.2.0; caching still deferred)"
python3 -m mdlm_engine.bench.harness \
    --adapter llada \
    --model_path "$LLADA_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --max_new_tokens 512 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --out "$RESULTS_DIR/07_llada_smoke.json" 2>&1 | \
    tee "$RESULTS_DIR/07_llada_smoke.log"
echo

# ---- 8. Speedup summary ----
echo "[8/8] Speedup summary"
RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/08_summary.log"
import json, os
results_dir = os.environ["RESULTS_DIR"]
with open(f"{results_dir}/03_dream_pathB.json") as f: b = json.load(f)
with open(f"{results_dir}/04_dream_pathA.json") as f: a = json.load(f)
print()
print(f"{'metric':25s}  {'PATH B (baseline)':>20s}  {'PATH A (v0.2.1)':>20s}  {'delta':>10s}")
print("-" * 80)
def row(label, key):
    bv, av = b[key], a[key]
    delta = (bv / av) if 'sec' not in key.lower() and 'forwards' not in key.lower() else (bv / av if av else 0)
    if 's_per_problem' in key or 'wall' in key or 'forwards' in key:
        delta = f"{bv/av:.2f}x faster" if av else "n/a"
    else:
        delta = f"{(av-bv)*100:+.1f}pp" if isinstance(av, float) and av < 1.5 else f"{av-bv:+.1f}"
    print(f"{label:25s}  {bv!s:>20s}  {av!s:>20s}  {delta:>10s}")
row("pass@1 single-shot", "pass_at_1")
row("s/problem", "s_per_problem")
row("tokens/sec", "tokens_per_sec")
row("total forwards", "total_forwards")
row("peak VRAM (GB)", "peak_vram_gb")
PY
echo

# ---- Summary ----
echo "============================================================"
echo "Phase 2.1 artifacts: $RESULTS_DIR"
echo "============================================================"
ls -lh "$RESULTS_DIR"
echo
echo "v0.2.1 gates:"
echo "  PATH A verification          MUST pass (fast_dllm overlay must engage)"
echo "  Dream pass@1 PATH A          ≥ 0.85 (no quality regression vs PATH B)"
echo "  Dream s/problem PATH A       ≤ 6.0  (≥ 1.5x speedup vs ~12 baseline)"
echo "  Dream s/problem ratio (B/A)  ≥ 1.5  (the actual speedup measure)"
echo "  Cache equivalence (block≡dkv) PASS"
echo "  NaN count on PATH A          0 / 100"
echo "  LLaDA pass@1                 ≥ 0.48 (unchanged)"
echo
echo "If PATH A pass@1 regressed >5pp vs PATH B, the cache wiring is"
echo "corrupting outputs — inspect 05_cache_equivalence_pathA.log."
