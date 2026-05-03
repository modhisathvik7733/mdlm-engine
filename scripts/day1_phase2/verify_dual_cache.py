"""Phase 2 Day-1 spike — does the HF Hub Dream-Coder model accept fast_dllm's
extension kwargs (`dual_cache`, `replace_position`)?

Phase 2 wires our `DiffusionCache` into the model via
``model.forward(past_key_values=..., dual_cache=True, replace_position=...)``.
That call SIGNATURE is fast_dllm's extension to Dream's stock modeling code
(verified at `Dream-Coder/instruct/src/inference/fast_dllm/modeling_dream.py:451-526`).

If the model loaded from `Dream-org/Dream-Coder-v0-Instruct-7B` already has those
patches baked in (= fast_dllm's modeling files were upstreamed), Phase 2 plumbs
them straight through. If not, we need a fallback path (`use_cache=False` with
a clear warning) for stock-HF model loads.

This script answers that question in ~30 seconds (just inspects `forward`'s
signature; no actual generation needed).

Usage:
    python3 scripts/day1_phase2/verify_dual_cache.py \\
        --dream_path Dream-org/Dream-Coder-v0-Instruct-7B
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dream_path",
        default="Dream-org/Dream-Coder-v0-Instruct-7B",
        help="HF repo id or local path of a Dream-architecture model.",
    )
    ap.add_argument(
        "--llada_path",
        default="GSAI-ML/LLaDA-8B-Base",
        help="HF repo id or local path of LLaDA. Optional; if missing the LLaDA "
             "section of the spike is skipped.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/day1_phase2/dual_cache_support.json"),
    )
    ap.add_argument(
        "--skip_llada",
        action="store_true",
        help="Skip LLaDA inspection (useful if you only have one model loaded).",
    )
    args = ap.parse_args()

    findings: dict = {}

    # -------------------------------------------------------------- Dream
    print(f"Loading {args.dream_path} (config + modeling code only) ...")
    try:
        # We don't need the model weights — config + AutoModel registry is enough
        # to inspect the forward signature.
        from transformers import AutoConfig, AutoModel  # type: ignore

        cfg = AutoConfig.from_pretrained(args.dream_path, trust_remote_code=True)
        model_cls = AutoModel._model_mapping[type(cfg)]  # type: ignore[attr-defined]
        sig = inspect.signature(model_cls.forward)
    except Exception as e:  # noqa: BLE001
        # Fallback: actually load the model. Slower but always works.
        import torch
        from transformers import AutoModel

        print(f"  AutoModel mapping failed ({e}); loading weights to read forward signature.")
        m = AutoModel.from_pretrained(
            args.dream_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        sig = inspect.signature(m.forward)
        del m

    dream_params = list(sig.parameters.keys())
    findings["dream"] = {
        "path": args.dream_path,
        "forward_signature": dream_params,
        "accepts_dual_cache": "dual_cache" in dream_params,
        "accepts_replace_position": "replace_position" in dream_params,
        "accepts_past_key_values": "past_key_values" in dream_params,
        "accepts_use_cache": "use_cache" in dream_params,
    }
    print(f"  Dream forward params: {dream_params}")
    print(f"  dual_cache:        {'YES' if findings['dream']['accepts_dual_cache'] else 'NO'}")
    print(f"  replace_position:  {'YES' if findings['dream']['accepts_replace_position'] else 'NO'}")
    print(f"  past_key_values:   {'YES' if findings['dream']['accepts_past_key_values'] else 'NO'}")
    print(f"  use_cache:         {'YES' if findings['dream']['accepts_use_cache'] else 'NO'}")

    # -------------------------------------------------------------- LLaDA
    if not args.skip_llada:
        print(f"\nLoading {args.llada_path} ...")
        try:
            from transformers import AutoConfig, AutoModel  # type: ignore

            cfg = AutoConfig.from_pretrained(args.llada_path, trust_remote_code=True)
            model_cls = AutoModel._model_mapping[type(cfg)]  # type: ignore[attr-defined]
            sig = inspect.signature(model_cls.forward)
        except Exception as e:  # noqa: BLE001
            import torch
            from transformers import AutoModel

            print(f"  AutoModel mapping failed ({e}); loading weights to read forward signature.")
            m = AutoModel.from_pretrained(
                args.llada_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            sig = inspect.signature(m.forward)
            del m

        llada_params = list(sig.parameters.keys())
        findings["llada"] = {
            "path": args.llada_path,
            "forward_signature": llada_params,
            "accepts_dual_cache": "dual_cache" in llada_params,
            "accepts_replace_position": "replace_position" in llada_params,
            "accepts_past_key_values": "past_key_values" in llada_params,
            "accepts_use_cache": "use_cache" in llada_params,
        }
        print(f"  LLaDA forward params: {llada_params}")
        print(f"  dual_cache:        {'YES' if findings['llada']['accepts_dual_cache'] else 'NO'}")
        print(f"  past_key_values:   {'YES' if findings['llada']['accepts_past_key_values'] else 'NO'}")

    # -------------------------------------------------------------- Verdict
    print("\n" + "=" * 60)
    dream_supports = (
        findings["dream"]["accepts_past_key_values"]
        and findings["dream"]["accepts_use_cache"]
    )
    dream_full_dual_cache = (
        dream_supports
        and findings["dream"]["accepts_dual_cache"]
        and findings["dream"]["accepts_replace_position"]
    )

    if dream_full_dual_cache:
        verdict = (
            "PATH A — full dual_cache support. Phase 2 Dream wiring uses "
            "model.forward(past_key_values=..., dual_cache=True, "
            "replace_position=..., use_cache=True) directly."
        )
    elif dream_supports:
        verdict = (
            "PATH B — past_key_values supported but dual_cache extension missing. "
            "Stock HF caching is APPEND-ONLY (torch.cat([past, new], dim=-2)) and "
            "CANNOT accelerate masked diffusion (which passes the full sequence "
            "each step and needs in-place K/V replace at masked positions). "
            "DreamAdapter collapses PATH B → PATH C with a one-time warning. "
            "For the v0.2.0 speedup, load Dream via "
            "`mdlm_engine.models.dream_fastdllm.load_dream_fastdllm()` "
            "(or the bench harness flag --use_fastdllm_modeling)."
        )
    else:
        verdict = (
            "PATH C — Dream model does NOT accept past_key_values/use_cache. "
            "Phase 2 is BLOCKED on Dream. Options: (1) require fast_dllm-patched "
            "modeling files, (2) skip Dream caching and ship only LLaDA's "
            "(if it supports it), (3) defer Phase 2 entirely."
        )

    findings["verdict"] = verdict
    print(verdict)
    print("=" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2))
    print(f"\nFull findings: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
