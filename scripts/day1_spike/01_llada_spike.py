"""Day-1 Spike #1 — LLaDA convention discovery.

Goal: in ~2 hours, verify that LLaDA fits our `ModelAdapter` ABC. If it
doesn't, we revise the ABC before writing the engine.

The ABC requires the adapter to be able to answer:
    - mask_token_id (int)
    - pad_token_id (int)
    - eos_token_ids (list[int])
    - apply_chat_template(messages) -> Tensor
    - build_position_ids(input_ids, attn_1d) -> Tensor | None
    - build_attention_mask(attn_1d, seq_len) -> Tensor | str
    - shift_logits(logits) -> Tensor
    - forward(input_ids, attn_mask, position_ids, cache, use_cache) -> AdapterOutput

Discoverable from the model itself:
    - config.model_type
    - tokenizer's `<|mask|>`-equivalent
    - forward() signature
    - whether logits need shifting (test by argmax of one prompt)

Output: JSON with what we found. The architecture freezes on this evidence.

Reference: arxiv 2502.09992 (LLaDA paper) for the design + chat template.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llada_path",
        default="GSAI-ML/LLaDA-8B-Base",
        help="HF repo id or local path. LLaDA-8B-Base is the easiest to access; "
        "LLaDA-8B-Instruct may have a chat template we want to inspect too.",
    )
    ap.add_argument("--out", type=Path, default=Path("scripts/day1_spike/01_llada_spike.json"))
    args = ap.parse_args()

    findings: dict = {"llada_path": args.llada_path}

    # ----- 1. Tokenizer + special tokens -----
    print(f"Loading tokenizer from {args.llada_path} ...")
    tok = AutoTokenizer.from_pretrained(args.llada_path, trust_remote_code=True)

    # LLaDA's mask token: try common spellings.
    mask_candidates = ["<|mask|>", "<mask>", "[MASK]", "<|MASK|>", "<|mdm_mask|>"]
    mask_token_id = None
    mask_token_str = None
    for candidate in mask_candidates:
        tid = tok.convert_tokens_to_ids(candidate)
        if tid is not None and tid != tok.unk_token_id:
            mask_token_id = tid
            mask_token_str = candidate
            break
    if mask_token_id is None:
        # Last resort: scan added_tokens.
        added = tok.get_added_vocab()
        for s, tid in added.items():
            if "mask" in s.lower():
                mask_token_id = tid
                mask_token_str = s
                break

    findings["mask_token"] = {"str": mask_token_str, "id": mask_token_id}
    findings["pad_token"] = {"str": tok.pad_token, "id": tok.pad_token_id}
    findings["eos_token"] = {"str": tok.eos_token, "id": tok.eos_token_id}
    findings["vocab_size"] = tok.vocab_size

    # ----- 2. Chat template -----
    has_chat_template = tok.chat_template is not None
    findings["has_chat_template"] = has_chat_template
    if has_chat_template:
        sample = tok.apply_chat_template(
            [{"role": "user", "content": "Write add(a, b)."}],
            add_generation_prompt=True,
            tokenize=False,
        )
        findings["chat_template_sample"] = sample[:300]
        findings["chat_template_first_50_token_ids"] = tok(
            sample, add_special_tokens=False
        ).input_ids[:50]

    # ----- 3. Model config -----
    print(f"Loading model from {args.llada_path} (this is the slow step) ...")
    model = AutoModel.from_pretrained(
        args.llada_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()

    cfg = model.config
    findings["model_type"] = getattr(cfg, "model_type", "unknown")
    findings["hidden_size"] = getattr(cfg, "hidden_size", None)
    findings["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
    findings["num_attention_heads"] = getattr(cfg, "num_attention_heads", None)
    findings["num_key_value_heads"] = getattr(cfg, "num_key_value_heads", None)
    findings["head_dim"] = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads if cfg.hidden_size else None
    )

    # ----- 4. Forward signature -----
    sig = inspect.signature(model.forward)
    findings["forward_signature"] = list(sig.parameters.keys())
    # Does it accept attention_mask? position_ids? past_key_values?
    findings["forward_accepts"] = {
        "attention_mask": "attention_mask" in sig.parameters,
        "position_ids": "position_ids" in sig.parameters,
        "past_key_values": "past_key_values" in sig.parameters,
        "use_cache": "use_cache" in sig.parameters,
        "inputs_embeds": "inputs_embeds" in sig.parameters,
    }

    # ----- 5. Logit-shift discovery -----
    # Insert one mask token in a known position and see whether logits at
    # position i predict token at position i (no shift) or position i+1 (shift).
    print("Probing logit alignment ...")
    if mask_token_id is None:
        findings["logit_shift_test"] = {"skipped": "no mask_token_id"}
    else:
        # Build prompt: "def fib(n): return [MASK]"
        prompt = "def fib(n): return "
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        n_prompt = prompt_ids.shape[-1]

        # Append a single mask token, expect the model to predict at that position.
        seq = torch.cat(
            [prompt_ids, torch.tensor([[mask_token_id]], device="cuda")], dim=-1
        )
        with torch.no_grad():
            out = model(seq)
        logits = out.logits if hasattr(out, "logits") else out[0]  # [1, L, V]

        # Test "no shift": argmax at position n_prompt should be a sensible
        # next-token (e.g. "n" or "fib"). Test "shift right": argmax at
        # position n_prompt-1 (shifted to n_prompt) should be sensible.
        no_shift_argmax = int(logits[0, n_prompt].argmax())
        shift_right_argmax = int(logits[0, n_prompt - 1].argmax()) if n_prompt > 0 else None
        findings["logit_shift_test"] = {
            "prompt": prompt,
            "n_prompt_tokens": n_prompt,
            "argmax_at_pos_n_no_shift": no_shift_argmax,
            "decoded_no_shift": tok.decode([no_shift_argmax]),
            "argmax_at_pos_n_minus_1_shifted": shift_right_argmax,
            "decoded_shifted": tok.decode([shift_right_argmax])
            if shift_right_argmax is not None
            else None,
            "guidance": (
                "If `decoded_no_shift` looks like a plausible next token "
                "(e.g. 'n', '1', 'fib') then LLaDA does NOT need shift_logits "
                "(identity). If `decoded_shifted` looks plausible and "
                "`decoded_no_shift` doesn't, LLaDA needs the same shift Dream "
                "uses (cat([logits[:, :1], logits[:, :-1]], dim=1)). "
                "Inspect by hand and record the verdict in this JSON manually."
            ),
        }

    # ----- 6. ABC fitness verdict (filled in by hand after inspecting) -----
    findings["abc_verdict"] = {
        "_instructions": (
            "After inspecting the printed JSON, set this dict by hand to one of:\n"
            "  {'fits': true} — ABC unchanged, proceed to Day 2.\n"
            "  {'fits': false, 'needs_method': '<method_name>', 'why': '<reason>'} — "
            "revise ModelAdapter ABC before writing the engine."
        ),
        "fits": None,
    }

    # ----- 7. Output -----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nSpike output written to {args.out}\n")
    print("=" * 60)
    print("Findings summary:")
    print("=" * 60)
    print(f"  model_type: {findings['model_type']}")
    print(f"  mask_token: {findings['mask_token']}")
    print(f"  forward signature: {findings['forward_signature']}")
    print(f"  forward accepts: {findings['forward_accepts']}")
    print()
    print(f"Inspect {args.out}, set abc_verdict.fits by hand, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
