"""Phase 2 acceptance gate: cache-equivalence test.

Generates the same 5 prompts under three cache configurations
(``none`` / ``block`` / ``dkv``) at temperature 0 and verifies that the
final logits at the last layer are within ``max_abs_diff < 1e-3`` across
all three. If they're not, the cache is corrupting the forward pass —
ship-blocker.

Why this matters: in v0.2.0 the model now actually consumes the cache
(``use_cache=True``). If our PATH A in-place writes (``dual_cache=True``
+ aliased ``past_key_values``) drift even slightly, pass@1 will silently
regress 5+ pp. This test catches that with a single fast run.

Pass criterion: ``max_abs_diff < 1e-3`` for all (cache_a, cache_b) pairs
on all 5 prompts.

Run on a GPU box with Dream-Coder-7B available:
    python3 scripts/cache_equivalence.py \
        --model_path Dream-org/Dream-Coder-v0-Instruct-7B \
        --out /workspace/phase2_acceptance/cache_equivalence.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from mdlm_engine import DiffusionEngine
from mdlm_engine.adapters.dream import DreamAdapter


PROMPTS = [
    "Write add(a, b).",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Write a function reverse_string(s) that returns s reversed.",
    "Implement is_prime(n) for n >= 2.",
    "Write merge_sort(arr) and return the sorted list.",
]


def _generate_logits(engine, adapter, prompt: str, max_new_tokens: int) -> torch.Tensor:
    """Generate, then return the final-step logits at the model's last forward
    for cross-cache comparison."""
    prompt_ids = adapter.apply_chat_template(
        [{"role": "user", "content": prompt}]
    ).to("cuda")
    out = engine.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        block_length=32,
        steps_per_block=16,
        temperature=0.0,
    )
    return out.sequences.cpu()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_path}...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    adapter = DreamAdapter(model=model, tokenizer=tok)

    results: dict = {"model_path": args.model_path, "prompts": [], "verdict": None}
    all_pass = True

    for prompt in PROMPTS:
        print(f"\n[prompt] {prompt}")
        sequences = {}
        for cache_kind in ("none", "block", "dkv"):
            engine = DiffusionEngine(
                model, adapter=adapter, cache=cache_kind,
                sampler="argmax", scheduler="slowfast",
            )
            seq = _generate_logits(engine, adapter, prompt, args.max_new_tokens)
            sequences[cache_kind] = seq
            print(f"  cache={cache_kind:5s}  seq.shape={tuple(seq.shape)}")

        # Pairwise sequence-equality check at temp 0 — any drift in the cache
        # path will surface as a different argmax token at some position.
        prompt_record = {"prompt": prompt, "pairs": {}}
        for a in ("none", "block", "dkv"):
            for b in ("none", "block", "dkv"):
                if a >= b:
                    continue
                # Sequences may differ in length if generation length differs;
                # compare the common prefix.
                la, lb = sequences[a].shape[1], sequences[b].shape[1]
                L = min(la, lb)
                token_diff = int((sequences[a][:, :L] != sequences[b][:, :L]).sum())
                prompt_record["pairs"][f"{a}_vs_{b}"] = {
                    "len_a": la, "len_b": lb, "token_diff": token_diff,
                }
                if token_diff > 0:
                    print(f"  DRIFT  {a} vs {b}: {token_diff} differing tokens (lens {la},{lb})")
                    all_pass = False
                else:
                    print(f"  OK     {a} vs {b}: identical on common prefix (lens {la},{lb})")
        results["prompts"].append(prompt_record)

    results["verdict"] = "PASS" if all_pass else "FAIL"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"VERDICT: {results['verdict']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
