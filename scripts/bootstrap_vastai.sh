#!/usr/bin/env bash
# Bootstrap a fresh vast.ai box (image: vastai/pytorch_cuda-13.0.2-auto) for
# mdlm-engine Phase-2 work. Idempotent — safe to re-run.
#
# Usage (on the vast.ai box, NOT the laptop):
#   cd /workspace
#   git clone https://github.com/modhisathvik7733/mdlm-engine.git
#   cd mdlm-engine
#   bash scripts/bootstrap_vastai.sh
#
# Tested target: 1x RTX 5090 (32 GB), CUDA 13.0, Ubuntu 22.04, ~75 GB disk.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "mdlm-engine vast.ai bootstrap"
echo "  workspace: $WORKSPACE"
echo "  repo:      $REPO_DIR"
echo "============================================================"

# ---- 0. Sanity: GPU + torch already in the image ----
echo
echo "[0/5] Verifying pre-installed CUDA stack"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | head -2
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible to torch'
print(f'  torch    {torch.__version__}')
print(f'  CUDA     {torch.version.cuda}')
print(f'  device   {torch.cuda.get_device_name(0)}')
print(f'  capa     sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
"

# ---- 1. Python deps ----
echo
echo "[1/5] Installing mdlm-engine + bench/test extras (editable)"
cd "$REPO_DIR"
pip install --break-system-packages --upgrade pip >/dev/null
pip install --break-system-packages -e ".[bench,test]"
pip install --break-system-packages "huggingface_hub[cli]" hf_transfer

# Pin transformers to the range pyproject.toml allows (4.46-4.49). If the image
# shipped a newer one, downgrade quietly.
python3 -c "
import transformers, packaging.version as v
ok = v.parse('4.46') <= v.parse(transformers.__version__) < v.parse('4.50')
print('  transformers', transformers.__version__, 'OK' if ok else 'NEEDS PIN')
" || pip install --break-system-packages "transformers>=4.46,<4.50"

# ---- 2. Disk-space pre-check ----
echo
echo "[2/5] Disk-space pre-check (need ~35 GB free for both models)"
df -h "$WORKSPACE" | tail -1
FREE_GB=$(df --output=avail -BG "$WORKSPACE" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 35 ]; then
    echo "  WARNING: only ${FREE_GB} GB free — model downloads may fail."
    echo "  Consider HF_HOME=/tmp or a larger volume."
fi

# ---- 3. Pre-download both models to the HF cache ----
echo
echo "[3/5] Pre-downloading models (~31 GB total) via hf_transfer"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$WORKSPACE/.hf_cache}"
mkdir -p "$HF_HOME"
echo "  HF_HOME=$HF_HOME"

for MODEL in "Dream-org/Dream-Coder-v0-Instruct-7B" "GSAI-ML/LLaDA-8B-Base"; do
    echo "  -> $MODEL"
    huggingface-cli download "$MODEL" >/dev/null
done

echo "  cache size:"
du -sh "$HF_HOME" | sed 's/^/    /'

# ---- 4. Smoke: pytest (fast, CPU + a couple GPU contract tests) ----
echo
echo "[4/5] Running pytest (CPU + GPU contract tests)"
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -20

# ---- 5. Phase-2 day-1 architecture spike ----
echo
echo "[5/5] Phase-2 day-1 spike: dual_cache support verdict"
python3 scripts/day1_phase2/verify_dual_cache.py \
    --dream_path Dream-org/Dream-Coder-v0-Instruct-7B \
    --llada_path GSAI-ML/LLaDA-8B-Base

echo
echo "============================================================"
echo "Bootstrap complete."
echo
echo "Next:"
echo "  1. Reproduce v4 baseline on this box (verifies hardware parity):"
echo "     bash scripts/phase1_acceptance.sh"
echo "  2. Read the dual_cache verdict above; pick Phase-2 PATH (A/B/C)"
echo "     per scripts/day1_phase2/README.md."
echo "============================================================"
