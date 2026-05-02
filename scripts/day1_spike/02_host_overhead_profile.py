"""Day-1 Spike #2 — Host overhead profile.

Goal: measure what fraction of wall time is spent in Python / framework code
vs actual GPU compute, on a vanilla 256-step Dream-Coder generation.

Decision rule (from plan §"Phase 1 Day-1 critical work"):
    host_fraction > 30%  →  torch.compile(reduce-overhead) + static padding
                            is LOAD-BEARING in Phase 1.
    host_fraction ≤ 30%  →  torch.compile is a Phase-1 nice-to-have.

Output: JSON with timings.

Note: at 512 forwards × ~50-200 μs Python launch overhead each, host time can
easily be 25-100 ms even when GPU compute is fast. On RTX 5090 this might
be the difference between 8 s/problem and 4 s/problem.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModel, AutoTokenizer


def _gpu_time_us(e) -> float:
    """Read GPU self-time off a profiler event, handling the PyTorch rename.

    PyTorch >= ~2.4 renamed `cuda_time_total` → `device_time_total`. Some
    nightly builds drop the legacy attribute entirely, so a getattr() with
    eager fallback like `getattr(e, 'device_time_total', e.cuda_time_total)`
    crashes — the fallback is evaluated even when the new attr exists.
    """
    if hasattr(e, "device_time_total"):
        return float(e.device_time_total)
    if hasattr(e, "cuda_time_total"):
        return float(e.cuda_time_total)
    return 0.0


def total_self_time_ms(prof: profile, device: str) -> float:
    """Sum self-time of all events on the given device. Returns milliseconds."""
    events = prof.key_averages()
    total_us = 0.0
    for e in events:
        if device == "cpu":
            total_us += float(e.cpu_time_total)
        elif device == "cuda":
            total_us += _gpu_time_us(e)
        else:
            raise ValueError(device)
    return total_us / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path or HF repo id of a Dream-architecture diffusion LM.")
    ap.add_argument("--prompt", default=(
        "Write a Python function `fib(n)` that returns the n-th Fibonacci "
        "number using memoization. Return only the function definition in a "
        "```python code block."
    ))
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("scripts/day1_spike/02_host_overhead_profile.json"))
    args = ap.parse_args()

    print(f"Loading {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()

    # ----- prompt -----
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            return_tensors="pt", return_dict=True, add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
    else:
        # Manual <|im_start|>...<|im_end|> for Dream-Coder if no template.
        prompt_text = (
            f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = tok(prompt_text, return_tensors="pt").to("cuda")
        input_ids = inp.input_ids
        attention_mask = inp.attention_mask

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

    # ----- warmup -----
    print("Warmup pass (not measured) ...")
    with torch.no_grad():
        _ = model.diffusion_generate(input_ids, **gen_kwargs)

    # ----- timed wall clock -----
    print("Wall-clock pass ...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model.diffusion_generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    wall_s = time.time() - t0
    print(f"  wall: {wall_s:.2f}s")

    # ----- profile -----
    print("Profiled pass (slower due to instrumentation; use for ratios only) ...")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        with record_function("diffusion_generate"):
            with torch.no_grad():
                _ = model.diffusion_generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize()

    cpu_ms = total_self_time_ms(prof, "cpu")
    gpu_ms = total_self_time_ms(prof, "cuda")
    # Heuristic: how much would `torch.compile(reduce-overhead)` realistically
    # save? Empirically that's roughly proportional to the CPU/GPU ratio:
    #   - cpu_ms << gpu_ms  → CPU never blocks GPU; compile saves <5%.
    #   - cpu_ms ≈ gpu_ms   → significant overlap; compile saves 10-20%.
    #   - cpu_ms > gpu_ms   → CPU is the bottleneck; compile saves 25%+.
    # We report the ratio plainly and let the human read it; the previous
    # "host_only = cpu - min(cpu, gpu)" formula collapsed to 0 whenever
    # CPU ≤ GPU, which is misleadingly clean.
    cpu_to_gpu_ratio = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
    # Approximate "fraction of wall time that's host work waiting on nothing":
    # if CPU > GPU, the excess is pure host stall; otherwise CPU work overlaps.
    host_only_ms = max(0.0, cpu_ms - gpu_ms)
    # Wall fraction: host_only / max_resource (the bottleneck).
    bottleneck_ms = max(cpu_ms, gpu_ms)
    host_fraction = host_only_ms / bottleneck_ms if bottleneck_ms > 0 else 0.0

    # Top 10 ops by CPU time and CUDA time (informational).
    top_cpu = sorted(prof.key_averages(), key=lambda e: e.cpu_time_total, reverse=True)[:10]
    top_cuda = sorted(
        prof.key_averages(),
        key=_gpu_time_us,
        reverse=True,
    )[:10]

    # Decision rule based on cpu/gpu ratio (the real signal, per plan):
    if cpu_to_gpu_ratio > 1.5:
        decision = "torch.compile + static padding is LOAD-BEARING in Phase 1 (CPU dominates)"
    elif cpu_to_gpu_ratio > 0.5:
        decision = "torch.compile is useful (~10-25% saving expected) — Phase 1 nice-to-have"
    else:
        decision = "torch.compile is marginal (<10% saving expected) — defer to Phase 2"

    findings: dict = {
        "model_path": args.model_path,
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "wall_seconds": wall_s,
        "profile_cpu_self_ms": cpu_ms,
        "profile_gpu_self_ms": gpu_ms,
        "cpu_to_gpu_ratio": cpu_to_gpu_ratio,
        "host_only_ms_estimate": host_only_ms,
        "host_fraction_pct": host_fraction * 100.0,
        "decision": decision,
        "top10_cpu_ops": [
            {"op": e.key, "cpu_time_ms": e.cpu_time_total / 1000.0,
             "count": e.count}
            for e in top_cpu
        ],
        "top10_cuda_ops": [
            {"op": e.key,
             "gpu_time_ms": _gpu_time_us(e) / 1000.0,
             "count": e.count}
            for e in top_cuda
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2))
    print(f"\nProfile written to {args.out}\n")
    print("=" * 60)
    print("Profile summary:")
    print("=" * 60)
    print(f"  wall:                   {wall_s:.2f} s")
    print(f"  profile CPU self time:  {cpu_ms:.1f} ms")
    print(f"  profile GPU self time:  {gpu_ms:.1f} ms")
    print(f"  cpu/gpu ratio:          {cpu_to_gpu_ratio:.2f}")
    print(f"  estimated host-only:    {host_only_ms:.1f} ms")
    print(f"  host fraction:          {host_fraction * 100:.1f}%")
    print(f"  → {findings['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
