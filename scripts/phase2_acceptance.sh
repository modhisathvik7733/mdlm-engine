#!/usr/bin/env bash
# Phase 2 (v0.2.0) acceptance-gate runner.
#
# Layered on top of Phase 1's gate. The new bars per the v2 plan §"Phase 2":
#   1. tests/                                                  all 116+2 GPU green
#   2. Phase-2 day-1 spike: Dream-Coder forward signature      PATH A or B
#   3. Dream s/problem (single-shot, HE+ subset)               ≤ 4.5  (was 9.9 in v0.1.0)
#   4. Dream pass@1 single-shot                                ≥ 0.85 (was 0.900)
#   5. Cache equivalence: none ≡ block ≡ dkv on 5 prompts      max_abs_diff < 1e-3
#   6. NaN-freedom (100 generations at temp 0)                 0 / 100
#   7. LLaDA portability smoke (still runs end-to-end)         pass@1 ≥ 0.48
#
# Run on a vast.ai box AFTER bootstrap_vastai.sh. ~30-45 min of GPU time.
#
# Usage:
#   cd /workspace/mdlm-engine
#   bash scripts/phase2_acceptance.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LLADA_PATH="${LLADA_PATH:-GSAI-ML/LLaDA-8B-Base}"
RESULTS_DIR="$WORKSPACE/phase2_acceptance"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "Phase 2 acceptance gate — mdlm-engine v0.2.0 (cache wiring)"
echo "============================================================"
echo

# ---- 0. Show env ----
echo "[env]"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
python3 -c "import torch; print(f'  torch {torch.__version__} CUDA {torch.version.cuda}')"
echo

# ---- 1. Test suite ----
echo "[1/7] Test suite (pytest tests/)"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/01_pytest.log" | tail -8
echo

# ---- 2. Day-1 spike: which PATH does the HF Hub Dream-Coder model take? ----
echo "[2/7] Phase-2 day-1 spike: dual_cache support verdict"
python3 scripts/day1_phase2/verify_dual_cache.py \
    --dream_path "$DREAM_PATH" \
    --llada_path "$LLADA_PATH" 2>&1 | tee "$RESULTS_DIR/02_dual_cache_spike.log"
echo

# ---- 3+4. Dream-Coder benchmark (the speedup test) ----
echo "[3-4/7] Dream-Coder benchmark with use_cache=True"
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
    --out "$RESULTS_DIR/03_dream_v020.json" 2>&1 | \
    tee "$RESULTS_DIR/03_dream_v020.log"
echo

# ---- 5. Cache-equivalence test (NEW v0.2.0 gate) ----
echo "[5/7] Cache equivalence: none vs block vs dkv on 5 prompts"
python3 scripts/cache_equivalence.py \
    --model_path "$DREAM_PATH" \
    --max_new_tokens 64 \
    --out "$RESULTS_DIR/05_cache_equivalence.json" 2>&1 | \
    tee "$RESULTS_DIR/05_cache_equivalence.log" || \
    echo "  WARNING: cache equivalence FAILED — investigate before tagging v0.2.0"
echo

# ---- 6. NaN-freedom (unchanged from Phase 1) ----
echo "[6/7] NaN-freedom (100 generations on Dream at temp 0)"
DREAM_PATH="$DREAM_PATH" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/06_nan_check.log"
import os, torch
from transformers import AutoModel, AutoTokenizer
from mdlm_engine import DiffusionEngine
from mdlm_engine.adapters.dream import DreamAdapter

path = os.environ["DREAM_PATH"]
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
adapter = DreamAdapter(model=model, tokenizer=tok)
print(f"  PATH: {adapter._caps.path} (A=fast_dllm, B=stock HF, C=no cache)")
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

# ---- 7. LLaDA portability smoke (unchanged: LLaDA deferred to v0.2.1) ----
echo "[7/7] LLaDA portability smoke (use_cache=False per v0.2.0 deferral)"
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

# ---- Summary ----
echo "============================================================"
echo "Phase 2 acceptance-gate artifacts in $RESULTS_DIR"
echo "============================================================"
ls -lh "$RESULTS_DIR"
echo
echo "v0.2.0 gate thresholds depend on the day-1 spike verdict:"
echo
echo "  PATH A (fast_dllm-patched modeling — speedup engaged):"
echo "    Dream s/problem      v0.1.0=9.9   v0.2.0 target ≤ 4.5  (~2× speedup)"
echo "    Dream pass@1         v0.1.0=0.900 v0.2.0 floor  ≥ 0.85"
echo
echo "  PATH B/C (stock HF or no caching — adapter falls back to v0.1.0):"
echo "    Dream s/problem      v0.1.0=9.9   v0.2.0 ≈ 9.9 (parity, no speedup)"
echo "    Dream pass@1         v0.1.0=0.900 v0.2.0 ≈ 0.900 (parity)"
echo "    Real speedup deferred to v0.2.1 (bundle fast_dllm modeling)."
echo
echo "  Always (regardless of path):"
echo "    LLaDA pass@1         v0.1.0=0.500 v0.2.0 floor  ≥ 0.48"
echo "    Cache equivalence    PASS — none/block/dkv must match at temp 0"
echo "    NaN count            0 / 100"
echo
echo "If pass@1 regressed >5pp on PATH A, cache wiring is corrupting outputs"
echo "— inspect 05_cache_equivalence.json and 06_nan_check.log."
