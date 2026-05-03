#!/usr/bin/env bash
# v0.3.0 Day-1 MXFP8 viability spike.
#
# Gates Lever A. Confirms three things on the user's actual stack
# (RTX 5090 + PyTorch 2.11 cu130 + torchao + fast_dllm-patched modeling):
#   1. torchao's quantize_(model, Float8DynamicActivationFloat8WeightConfig())
#      runs without crashing on Dream-Coder-Instruct-7B loaded via
#      load_dream_fastdllm() (i.e. with trust_remote_code modeling on top).
#   2. The post-quantize forward signature still exposes dual_cache and
#      replace_position (PATH A still works after quantization).
#   3. Logit max-abs-diff vs bf16 reference < 0.05. Above 0.5 means defer.
#
# Cost: ~10 min (loads the model twice — bf16 reference and quantized);
# ~$0.10 of vast.ai 5090 time.
#
# Pass criteria for moving to Lever A wiring:
#   ✓ quantize_ runs without exception
#   ✓ dual_cache + replace_position still in forward signature
#   ✓ logit max-abs-diff < 0.05 (gate); 0.05–0.5 = "review";  > 0.5 = defer

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DREAM_PATH="${DREAM_PATH:-Dream-org/Dream-Coder-v0-Instruct-7B}"
RESULTS_DIR="$WORKSPACE/v0_3_0_spike"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

echo "============================================================"
echo "v0.3.0 Day-1 MXFP8 viability spike"
echo "============================================================"
echo
echo "Stack:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | head -2
python3 -c "
import torch, transformers
print(f'  torch        {torch.__version__}')
print(f'  CUDA         {torch.version.cuda}')
print(f'  transformers {transformers.__version__}')
try:
    import torchao
    print(f'  torchao      {torchao.__version__}')
except ImportError:
    print('  torchao      NOT INSTALLED — run: pip install --break-system-packages torchao>=0.17.0')
    raise
"
echo

DREAM_PATH="$DREAM_PATH" RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY' 2>&1 | tee "$RESULTS_DIR/spike.log"
"""MXFP8 viability spike.

Loads Dream-Coder twice: once at bf16 (reference), once at bf16 + MXFP8
quantization. Runs a single forward on a fixed prompt. Compares logits.
"""
import json
import os
from pathlib import Path

import torch

DREAM_PATH = os.environ["DREAM_PATH"]
RESULTS_DIR = Path(os.environ["RESULTS_DIR"])

# Use a deterministic prompt; we want logit comparison to be reproducible.
PROMPT_TEXT = (
    "Complete the following Python function. Return only the full function "
    "definition in a ```python code block.\n\n"
    "```python\ndef add(a, b):\n    return\n```"
)

print("[1/4] Loading Dream-Coder via load_dream_fastdllm() at bf16 (reference) ...")
from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(DREAM_PATH, trust_remote_code=True)
model_bf16 = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cuda").eval()

# Build a prompt and run a forward. Match how DreamAdapter's apply_chat_template would.
chat = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT_TEXT}],
    return_tensors="pt", return_dict=True, add_generation_prompt=True,
)
input_ids = chat["input_ids"].to("cuda")
attn = chat.get("attention_mask")
attn = attn.to("cuda") if attn is not None else None

print(f"   input_ids.shape = {tuple(input_ids.shape)}")

with torch.inference_mode():
    out_bf16 = model_bf16(input_ids=input_ids, attention_mask=attn, use_cache=False)
logits_bf16 = (out_bf16.logits if hasattr(out_bf16, "logits") else out_bf16[0]).float()
print(f"   bf16 logits.shape = {tuple(logits_bf16.shape)}, dtype before .float() was {(out_bf16.logits if hasattr(out_bf16, 'logits') else out_bf16[0]).dtype}")

# Free the bf16 model — we have its logits saved.
del model_bf16, out_bf16
torch.cuda.empty_cache()
print()

print("[2/4] Loading Dream-Coder + applying MXFP8 quantization ...")
model_q = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cuda").eval()

import inspect
target = model_q._orig_mod if hasattr(model_q, "_orig_mod") else model_q
pre_sig = inspect.signature(target.forward)
pre_dual_cache = "dual_cache" in pre_sig.parameters
pre_replace_position = "replace_position" in pre_sig.parameters
print(f"   pre-quantize: dual_cache={pre_dual_cache}, replace_position={pre_replace_position}")

try:
    from torchao.quantization import (
        quantize_, Float8DynamicActivationFloat8WeightConfig,
    )
except ImportError as e:
    print(f"   FAIL: torchao import error: {e}")
    raise

try:
    quantize_(model_q, Float8DynamicActivationFloat8WeightConfig())
    print("   quantize_() returned without exception ✓")
except Exception as e:
    print(f"   FAIL: quantize_() raised {type(e).__name__}: {e}")
    raise
print()

print("[3/4] Verifying PATH A signature still intact post-quantize ...")
target = model_q._orig_mod if hasattr(model_q, "_orig_mod") else model_q
post_sig = inspect.signature(target.forward)
post_dual_cache = "dual_cache" in post_sig.parameters
post_replace_position = "replace_position" in post_sig.parameters
print(f"   post-quantize: dual_cache={post_dual_cache}, replace_position={post_replace_position}")
sig_ok = post_dual_cache and post_replace_position
print(f"   Signature preserved: {'YES ✓' if sig_ok else 'NO ✗'}")
print()

print("[4/4] Running quantized forward + comparing logits ...")
with torch.inference_mode():
    out_q = model_q(input_ids=input_ids, attention_mask=attn, use_cache=False)
logits_q = (out_q.logits if hasattr(out_q, "logits") else out_q[0]).float()
print(f"   quant logits.shape = {tuple(logits_q.shape)}")

# Logit comparison
diff = (logits_bf16 - logits_q).abs()
max_abs = float(diff.max())
mean_abs = float(diff.mean())
# Argmax agreement at every position
am_bf16 = logits_bf16.argmax(dim=-1)
am_q = logits_q.argmax(dim=-1)
argmax_agreement = float((am_bf16 == am_q).float().mean())

print(f"   max  |Δlogit| = {max_abs:.4f}")
print(f"   mean |Δlogit| = {mean_abs:.4f}")
print(f"   argmax agreement = {argmax_agreement:.4f}")
print()

# Verdict
verdict = "PASS"
gate_reason = ""
if not sig_ok:
    verdict = "FAIL"
    gate_reason = "PATH A signature lost after quantize_"
elif max_abs >= 0.5:
    verdict = "FAIL"
    gate_reason = f"Logit drift {max_abs:.3f} >= 0.5 (defer threshold)"
elif max_abs >= 0.05:
    verdict = "REVIEW"
    gate_reason = f"Logit drift {max_abs:.3f} in [0.05, 0.5) — needs human review before shipping"
else:
    verdict = "PASS"
    gate_reason = f"Logit drift {max_abs:.3f} < 0.05; argmax agreement {argmax_agreement:.4f}"

result = {
    "stack": {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    },
    "model_path": DREAM_PATH,
    "input_ids_shape": list(input_ids.shape),
    "pre_quantize": {
        "dual_cache_in_sig": pre_dual_cache,
        "replace_position_in_sig": pre_replace_position,
    },
    "post_quantize": {
        "dual_cache_in_sig": post_dual_cache,
        "replace_position_in_sig": post_replace_position,
    },
    "logit_diff": {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "argmax_agreement": argmax_agreement,
    },
    "verdict": verdict,
    "gate_reason": gate_reason,
}
out_path = RESULTS_DIR / "spike.json"
out_path.write_text(json.dumps(result, indent=2))

print("=" * 60)
print(f"VERDICT: {verdict}")
print(f"   {gate_reason}")
print("=" * 60)
print(f"\nFull findings: {out_path}")

# Exit nonzero on FAIL so the shell sees it
if verdict == "FAIL":
    raise SystemExit(1)
PY
echo
echo "============================================================"
echo "Spike complete. Read $RESULTS_DIR/spike.json"
echo "============================================================"
echo
echo "Action based on verdict:"
echo "  PASS    → proceed to Lever A wiring (mdlm_engine/bench/harness.py)"
echo "  REVIEW  → eyeball the logit diff distribution; small drift may still"
echo "            be acceptable depending on argmax_agreement"
echo "  FAIL    → drop Lever A from v0.3.0; ship Levers B+C only"
