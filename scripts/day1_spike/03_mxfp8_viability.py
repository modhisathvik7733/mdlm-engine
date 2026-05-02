"""Day-1 Spike #3 — MXFP8 viability check.

Goal: in ~1 hour, determine whether `torchao.quantization.float8_dynamic_activation_float8_weight`
on a 7B Dream-Coder produces stable logits in the 32-step diffusion sampling
loop, or if the numerical drift is too large.

Decision rule (from plan §"Phase 1 Day-1 critical work"):
    max_abs_logit_diff < 0.05  →  MXFP8 ships in v0.1.0
    max_abs_logit_diff in [0.05, 0.5]  →  ship behind a flag with a warning
    max_abs_logit_diff > 0.5 OR any NaN  →  defer MXFP8 to Phase 2

References:
- PyTorch blog "Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO" (July 2025)
- torchao APIs: https://github.com/pytorch/ao
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", default=(
        "Write a Python function `fib(n)` that returns the n-th Fibonacci "
        "number. Return only the function in a ```python code block."
    ))
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("scripts/day1_spike/03_mxfp8_viability.json"))
    args = ap.parse_args()

    findings: dict = {"model_path": args.model_path, "steps": args.steps}

    # ----- load tokenizer + prompt -----
    print(f"Loading tokenizer from {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.chat_template:
        prompt_inputs = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            return_tensors="pt", return_dict=True, add_generation_prompt=True,
        )
    else:
        prompt_text = (
            f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        prompt_inputs = tok(prompt_text, return_tensors="pt", return_attention_mask=True)
    input_ids = prompt_inputs.input_ids.to("cuda")
    attention_mask = prompt_inputs.attention_mask.to("cuda")

    # ----- baseline (bf16) -----
    print("Loading bf16 baseline ...")
    bf16_model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    # Forward once at the prompt (no generation) — capture logits at the end of prompt.
    with torch.no_grad():
        bf16_logits = bf16_model(input_ids, attention_mask=attention_mask).logits.float().cpu()

    # ----- timed bf16 generation -----
    gen_kwargs = dict(
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=0.0,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        output_history=False,
        return_dict_in_generate=True,
    )

    print("Warmup + timed bf16 generation ...")
    with torch.no_grad():
        _ = bf16_model.diffusion_generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        bf16_out = bf16_model.diffusion_generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    bf16_wall = time.time() - t0
    bf16_seq = bf16_out.sequences[0].cpu().tolist()

    # Free bf16 before loading fp8.
    del bf16_model
    torch.cuda.empty_cache()

    # ----- fp8 quantized -----
    print("Loading model + quantizing to MXFP8 ...")
    try:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight  # type: ignore
    except Exception as e:  # noqa: BLE001
        findings["error"] = f"torchao import failed: {type(e).__name__}: {e}"
        findings["decision"] = "DEFER MXFP8 to Phase 2 (torchao not importable)"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(findings, indent=2))
        print(json.dumps(findings, indent=2))
        return 1

    fp8_model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    try:
        quantize_(fp8_model, float8_dynamic_activation_float8_weight())
    except Exception as e:  # noqa: BLE001
        findings["error"] = f"torchao quantize_ failed: {type(e).__name__}: {e}"
        findings["decision"] = "DEFER MXFP8 to Phase 2 (quantize call errored)"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(findings, indent=2))
        print(json.dumps(findings, indent=2))
        return 1

    # ----- prompt-only logit diff -----
    with torch.no_grad():
        fp8_logits = fp8_model(input_ids, attention_mask=attention_mask).logits.float().cpu()
    max_abs_diff = (bf16_logits - fp8_logits).abs().max().item()
    mean_abs_diff = (bf16_logits - fp8_logits).abs().mean().item()
    nan_in_logits = bool(torch.isnan(fp8_logits).any().item())

    findings["prompt_logit_diff"] = {
        "max_abs": max_abs_diff,
        "mean_abs": mean_abs_diff,
        "nan_in_fp8_logits": nan_in_logits,
    }

    # ----- timed fp8 generation -----
    print("Warmup + timed fp8 generation ...")
    with torch.no_grad():
        _ = fp8_model.diffusion_generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        with torch.no_grad():
            fp8_out = fp8_model.diffusion_generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize()
        fp8_wall = time.time() - t0
        fp8_seq = fp8_out.sequences[0].cpu().tolist()
        gen_error = None
    except Exception as e:  # noqa: BLE001
        fp8_wall = None
        fp8_seq = None
        gen_error = f"{type(e).__name__}: {e}"
    findings["bf16_wall_seconds"] = bf16_wall
    findings["fp8_wall_seconds"] = fp8_wall
    findings["fp8_generation_error"] = gen_error
    if fp8_wall:
        findings["fp8_speedup"] = bf16_wall / fp8_wall

    # ----- decoded comparison -----
    if fp8_seq is not None:
        prompt_len = input_ids.shape[-1]
        bf16_decoded = tok.decode(bf16_seq[prompt_len:], skip_special_tokens=True)
        fp8_decoded = tok.decode(fp8_seq[prompt_len:], skip_special_tokens=True)
        findings["bf16_decoded"] = bf16_decoded[:400]
        findings["fp8_decoded"] = fp8_decoded[:400]
        findings["decoded_match_first_100_tokens"] = (
            bf16_seq[prompt_len:prompt_len + 100] == fp8_seq[prompt_len:prompt_len + 100]
        )

    # ----- decision -----
    if gen_error or nan_in_logits:
        decision = "DEFER MXFP8 to Phase 2 (NaN or generation error)"
    elif max_abs_diff < 0.05:
        decision = "SHIP MXFP8 in v0.1.0 (logit drift < 0.05)"
    elif max_abs_diff < 0.5:
        decision = (
            "SHIP MXFP8 behind --quant mxfp8 flag, default off, "
            "with NaN-at-low-temp warning"
        )
    else:
        decision = "DEFER MXFP8 to Phase 2 (logit drift > 0.5)"
    findings["decision"] = decision

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2))
    print("\n" + "=" * 60)
    print("MXFP8 viability summary:")
    print("=" * 60)
    print(f"  bf16 wall:              {bf16_wall:.2f} s")
    print(f"  fp8 wall:               {fp8_wall and f'{fp8_wall:.2f} s' or 'errored'}")
    if fp8_wall:
        print(f"  fp8 speedup:            {bf16_wall / fp8_wall:.2f}×")
    print(f"  prompt-logit max-abs:   {max_abs_diff:.4f}")
    print(f"  prompt-logit mean-abs:  {mean_abs_diff:.4f}")
    print(f"  NaN in fp8 logits:      {nan_in_logits}")
    print(f"  → {decision}")
    print(f"\nFull findings: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
