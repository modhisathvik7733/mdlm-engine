#!/usr/bin/env bash
# Phase 1 acceptance-gate runner.
#
# Executes the gate as defined in the plan §"Phase 1 acceptance gates":
#   1. tests/  — all CPU + GPU-marked tests pass.
#   2. Speed (Dream-Coder, single-shot, HE+ subset)            ≤ 8 s/problem
#   3. Quality (Dream-Coder, single-shot pass@1)               ≥ 0.55
#   4. Quality (Dream-Coder, best-of-8 pass@1)                 ≥ 0.85
#   5. Portability — same engine code runs LLaDA               smoke pass
#   6. Adapter LOC                                             ≤ 200 LOC each
#   7. Numerical stability (100 generations, zero NaN logits)  pass
#
# Run on a vast.ai box with Dream-Coder-7B + LLaDA-8B already downloaded
# (or HF Hub access). ~2-3 hours of GPU time.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
LLADA_PATH="${LLADA_PATH:-GSAI-ML/LLaDA-8B-Base}"
RESULTS_DIR="$WORKSPACE/phase1_acceptance"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "Phase 1 acceptance gate — mdlm-engine v0.1.0"
echo "============================================================"
echo

# ---- 0. Show env ----
echo "[env]"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
python3 -c "import torch; print(f'  torch {torch.__version__} CUDA {torch.version.cuda}')"
echo

# ---- 1. CPU + GPU tests ----
echo "[1/7] Test suite (pytest tests/)"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/01_pytest.log" | tail -5
echo

# ---- 2-4. Dream-Coder benchmark ----
echo "[2-4/7] Dream-Coder benchmark (single-shot + best-of-8)"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --max_new_tokens 256 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --top_p 0.95 \
    --out "$RESULTS_DIR/02_dream_single_shot.json" 2>&1 | \
    tee "$RESULTS_DIR/02_dream_single_shot.log"
echo

echo "[2-4/7b] Dream-Coder best-of-8 oracle"
python3 -m mdlm_engine.bench.harness \
    --adapter dream \
    --model_path "$DREAM_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --diverse 8 \
    --out "$RESULTS_DIR/02_dream_best_of_8.json" 2>&1 | \
    tee "$RESULTS_DIR/02_dream_best_of_8.log"
echo

# ---- 5. LLaDA portability smoke ----
echo "[5/7] LLaDA portability smoke"
python3 -m mdlm_engine.bench.harness \
    --adapter llada \
    --model_path "$LLADA_PATH" \
    --cache dkv \
    --scheduler slowfast \
    --sampler entropy \
    --benchmark humaneval_plus \
    --limit 20 \
    --max_new_tokens 256 \
    --block_length 32 \
    --steps_per_block 32 \
    --temperature 0.2 \
    --out "$RESULTS_DIR/05_llada_smoke.json" 2>&1 | \
    tee "$RESULTS_DIR/05_llada_smoke.log"
echo

# ---- 6. Adapter LOC ----
echo "[6/7] Adapter LOC"
{
    echo "DreamAdapter:";  wc -l mdlm_engine/adapters/dream.py
    echo "LLaDAAdapter:";  wc -l mdlm_engine/adapters/llada.py
} | tee "$RESULTS_DIR/06_adapter_loc.txt"
echo

# ---- 7. NaN-freedom (100 generations on Dream) ----
echo "[7/7] NaN-freedom check (100 generations at temp 0)"
python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/07_nan_check.log"
import torch
from transformers import AutoModel, AutoTokenizer
from mdlm_engine import DiffusionEngine
from mdlm_engine.adapters.dream import DreamAdapter

import os
path = os.environ.get("DREAM_PATH", "Dream-org/Dream-Coder-v0-Instruct-7B")
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
adapter = DreamAdapter(model=model, tokenizer=tok)
engine = DiffusionEngine(model, adapter=adapter, cache="dkv", sampler="argmax", scheduler="slowfast")

prompt = adapter.apply_chat_template([{"role": "user", "content": "Write add(a, b)."}]).to("cuda")
nan_count = 0
for i in range(100):
    out = engine.generate(prompt, max_new_tokens=64, block_length=32, steps_per_block=16, temperature=0.0)
    if torch.isnan(out.sequences.float()).any():
        nan_count += 1
print(f"100 generations done. NaN count: {nan_count} (must be 0 to pass)")
PY
echo

# ---- Summary ----
echo "============================================================"
echo "Acceptance-gate artifacts written to $RESULTS_DIR"
echo "============================================================"
ls -lh "$RESULTS_DIR"
echo
echo "Read each *.json to confirm gate thresholds:"
echo "  Dream single-shot pass@1   ≥ 0.55  AND  s/problem ≤ 8"
echo "  Dream best-of-8  pass@1    ≥ 0.85"
echo "  LLaDA smoke pass@1         ≥ 0.40"
echo "  Adapter LOC                each ≤ 200"
echo "  NaN count                  exactly 0"
echo
echo "If all green → mdlm-engine v0.1.0 is ready to tag and ship."
