"""Bench harness — runs the Phase-1 acceptance gate.

CLI:

    python -m mdlm_engine.bench.harness \\
        --adapter dream \\
        --model_path Dream-org/Dream-Coder-v0-Instruct-7B \\
        --cache dkv --scheduler slowfast --sampler entropy \\
        --benchmark humaneval_plus --limit 20 \\
        --report single-shot --report best-of-8 \\
        --report tokens-per-sec --report peak-vram

Reports:
    s/problem, tokens/sec, num_forwards, peak VRAM,
    pass@1 (single-shot), pass@1 (best-of-N oracle).

Per the v2 plan, this is the canonical bench used by all acceptance
gates. Distinguishes single-shot pass@1 (real) from best-of-N (oracle)
explicitly so we don't mistake one for the other.

Phase 1 day 7: harness wired up; the actual model load + generate runs
on a GPU box at the day-8 acceptance gate. Without GPU, `--limit 0`
exits cleanly after harness self-test.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Regex that pulls the first/longest python block out of an LLM completion.
# Same pattern used in the user's existing eval_diverse_fastdllm.py.
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """One run's metrics. Serializable to JSON for cross-version diff."""

    adapter: str
    model_path: str
    cache: str
    scheduler: str
    sampler: str
    benchmark: str
    limit: int
    n_problems_run: int = 0
    pass_at_1_single_shot: float = 0.0
    pass_at_1_best_of_n: float = 0.0
    seconds_per_problem: float = 0.0
    tokens_per_second: float = 0.0
    total_forwards: int = 0
    peak_vram_gb: float = 0.0
    wall_seconds: float = 0.0
    n_diverse_configs: int = 1
    notes: list[str] = field(default_factory=list)
    per_problem: list[dict] = field(default_factory=list)
    # Per-problem records: {"task_id": str, "passed": bool, "seconds": float,
    # "completion_len": int}. Lets v0.2.2-style ablations diff which problems
    # regressed between two configs.


# ---------------------------------------------------------------------------
# Eight diverse configs (matches the user's existing
# diffucoder-7b-cpgrpo/eval_amplified_diverse_fast.py shape; see plan §
# "Best-of-N math" for the metric semantics)
# ---------------------------------------------------------------------------


DIVERSE_CONFIGS = [
    # (sampler, scheduler, temperature, top_p, steps_per_block)
    ("entropy",      "slowfast",   0.2, 0.95, 32),
    ("entropy",      "confidence", 0.5, 0.95, 32),
    ("entropy",      "uniform",    0.7, 0.95, 16),
    ("entropy",      "slowfast",   0.9, 0.92, 16),
    ("maskgit_plus", "slowfast",   0.4, 0.95, 32),
    ("maskgit_plus", "uniform",    0.7, 0.95, 16),
    ("topk_margin",  "slowfast",   0.4, 0.95, 32),
    ("topk_margin",  "uniform",    0.7, 0.95, 16),
]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="dream", help="model_type for adapter registry")
    ap.add_argument("--model_path", default="Dream-org/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--cache", default="dkv", choices=["none", "block", "dkv"])
    ap.add_argument("--scheduler", default="slowfast", choices=["uniform", "confidence", "slowfast"])
    ap.add_argument("--sampler", default="entropy",
                    choices=["argmax", "maskgit_plus", "entropy", "margin", "topk_margin"])
    ap.add_argument("--benchmark", default="humaneval_plus", choices=["humaneval_plus"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="Default 512 (v0.2.2): pass@1 0.6707 single-shot full HE+ on Dream-Coder. "
                         "256 is faster (~9 s/problem) but caps ~30%% of HE+ problems mid-function "
                         "(pass@1 drops to 0.54). 768 is the paper-default; ~3pp more for 1.5x cost.")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps_per_block", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--diverse", type=int, default=0,
                    help="If >0, run best-of-N over the first N DIVERSE_CONFIGS")
    ap.add_argument("--speculative_k", type=int, default=1,
                    help="v0.3.0 self-speculative decoding (arxiv 2510.04147). "
                         "After each step, propose extra high-confidence "
                         "masked positions and verify with one extra forward. "
                         "Default 1 (v0.3.0 ships SSD ON). Pass 0 to disable "
                         "and get v0.2.2 quality (0.6707 pass@1 / 11.6s "
                         "vs v0.3.0's 0.6524 / 6.4s on full HE+).")
    ap.add_argument("--speculative_threshold", type=float, default=0.99,
                    help="Min top-1 softmax probability for SSD to propose. "
                         "Default 0.99 (v0.3.0): near-lossless at production "
                         "sampling settings (-1.83 pp pass@1 measured at full "
                         "HE+, 1.80x speedup). Pass 0.95 for the speed preset "
                         "(2.02x speedup, -4.87 pp).")
    ap.add_argument("--speculative_tree_k", type=int, default=1,
                    help="v0.4.0 tree speculative decoding. 1 (default) keeps "
                         "v0.3.0 single-branch SSD bit-identical. 2 enables a "
                         "second verify forward over positions in "
                         "[speculative_band_low, speculative_threshold). "
                         "Lossless by argmax-match on borrowed branch-0 "
                         "context.")
    ap.add_argument("--speculative_band_low", type=float, default=0.97,
                    help="Tree-spec branch-1 lower confidence bound (default "
                         "0.97). Only used when --speculative_tree_k > 1.")
    ap.add_argument("--speculative_max_proposals_branch_1", type=int, default=4,
                    help="Cap on tree-spec branch-1 proposals per step "
                         "(default 4). Bounds branch-1 verify-forward cost.")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile on the model")
    ap.add_argument("--quant", default="", choices=["", "mxfp8", "int8", "int4"])
    ap.add_argument("--use_fastdllm_modeling", action="store_true",
                    help="Dream only: load with fast_dllm-patched modeling (PATH A; ~2x speedup). "
                         "Overlays mdlm_engine/models/dream_fastdllm/modeling_dream.py onto the "
                         "HF cache copy at load time.")
    ap.add_argument("--out", type=Path, default=Path("bench_results.json"))
    ap.add_argument("--no_run", action="store_true",
                    help="Skip actual generation (CPU self-test only)")
    args = ap.parse_args(argv)

    result = BenchResult(
        adapter=args.adapter,
        model_path=args.model_path,
        cache=args.cache,
        scheduler=args.scheduler,
        sampler=args.sampler,
        benchmark=args.benchmark,
        limit=args.limit,
        n_diverse_configs=max(args.diverse, 1),
    )

    if args.no_run:
        result.notes.append("--no_run: harness self-test only, no model loaded")
        _write(result, args.out)
        print(f"Self-test OK. Result schema written to {args.out}")
        return 0

    # GPU path — only invoked when actually running on a real box.
    return _run_benchmark(args, result)


def _run_benchmark(args, result: BenchResult) -> int:
    """Real benchmark run. Lives in a function (not main) so the import
    of torch+evalplus only happens when the user actually wants to run."""
    try:
        import torch  # noqa: F401
    except ImportError:
        result.notes.append("torch not installed; cannot run")
        _write(result, args.out)
        return 1

    if not torch.cuda.is_available():
        result.notes.append("CUDA unavailable; this benchmark requires a GPU")
        _write(result, args.out)
        print("Cannot run without CUDA. Use --no_run for a self-test.")
        return 1

    # Lazy imports — keeps `import mdlm_engine.bench.harness` cheap.
    from transformers import AutoModel, AutoTokenizer

    from mdlm_engine import DiffusionEngine
    from mdlm_engine.adapters import get_adapter_for
    from mdlm_engine.ops.compile import maybe_compile_model

    print(f"Loading {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.use_fastdllm_modeling:
        if args.adapter != "dream":
            print(f"  WARNING: --use_fastdllm_modeling is dream-only; ignoring for adapter={args.adapter}")
            model = AutoModel.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to("cuda").eval()
        else:
            from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm
            print("  using fast_dllm-patched modeling_dream.py (PATH A)")
            model = load_dream_fastdllm(
                args.model_path, torch_dtype=torch.bfloat16,
            ).to("cuda").eval()
    else:
        model = AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to("cuda").eval()

    # v0.3.0 Lever A: MXFP8 quantization via torchao. Apply AFTER the model
    # is on CUDA in eval mode — torchao's quantize_ operates on the live
    # nn.Module and replaces linear layer weights in-place with FP8.
    # Day-1 viability spike (scripts/v0_3_0_spike.sh) gates this; ship only
    # if logit max-abs-diff vs bf16 < 0.05.
    if args.quant == "mxfp8":
        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig, quantize_,
            )
        except ImportError as e:
            raise SystemExit(
                f"--quant mxfp8 requires torchao>=0.17.0; install with "
                f"`pip install --break-system-packages 'torchao>=0.17.0'`. "
                f"Original error: {e}"
            )
        print("  applying MXFP8 quantization via torchao ...")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        print("  quantize_() complete (linear weights now FP8).")
    elif args.quant in ("int8", "int4"):
        # Phase 1 viability spike showed int8/int4 NaN'd on Blackwell + nightly.
        # Refuse loudly rather than silently doing nothing.
        raise SystemExit(
            f"--quant {args.quant} is documented broken on RTX 5090 + "
            f"PyTorch nightly cu128/cu130 (NaN cascades). Use mxfp8 instead."
        )

    model = maybe_compile_model(model, enabled=args.compile)

    adapter_cls = get_adapter_for(args.adapter)
    adapter = adapter_cls(model=model, tokenizer=tok)

    # Load HumanEval+ — only path we ship in Phase 1.
    if args.benchmark == "humaneval_plus":
        from evalplus.data import get_human_eval_plus
        items = list(get_human_eval_plus().items())
        if args.limit:
            items = items[: args.limit]
    else:
        result.notes.append(f"unknown benchmark: {args.benchmark}")
        _write(result, args.out)
        return 1

    # Diverse best-of-N or single-shot dispatch.
    diverse_n = args.diverse if args.diverse > 0 else 0
    if diverse_n > len(DIVERSE_CONFIGS):
        print(f"  --diverse {diverse_n} > len(DIVERSE_CONFIGS)={len(DIVERSE_CONFIGS)}; clamping")
        diverse_n = len(DIVERSE_CONFIGS)

    if diverse_n > 0:
        # Build N engines (one per config). Engines are cheap — they share
        # the loaded model and adapter; only sampler_fn / scheduler_fn /
        # cache_kind differ.
        engines = []
        for sampler_name, scheduler_name, _temp, _top_p, _steps in DIVERSE_CONFIGS[:diverse_n]:
            eng = DiffusionEngine(
                model, adapter=adapter,
                cache=args.cache, sampler=sampler_name, scheduler=scheduler_name,
            )
            engines.append(eng)
        print(f"  diverse best-of-{diverse_n}: 8 configs (sampler/scheduler/temp/top_p/steps)")
        for j, (s, sch, t_, p_, st) in enumerate(DIVERSE_CONFIGS[:diverse_n]):
            print(f"    [{j}] sampler={s:13s} scheduler={sch:11s} temp={t_:.1f} top_p={p_:.2f} steps={st}")
    else:
        engine = DiffusionEngine(
            model, adapter=adapter,
            cache=args.cache, sampler=args.sampler, scheduler=args.scheduler,
        )

    n_pass = 0
    total_tokens = 0
    total_forwards = 0
    total_attempts = 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for i, (task_id, row) in enumerate(items):
        t_problem_start = time.time()
        prompt_ids = adapter.apply_chat_template(
            [{"role": "user", "content": _format_prompt(row['prompt'])}],
        ).to("cuda")

        if diverse_n > 0:
            # First-pass-acceptance best-of-N: try configs in order, stop at
            # first success. Records n_attempts per problem so cost can be
            # amortized properly (not all problems need all N).
            passed = False
            n_attempts_this = 0
            last_decoded = ""
            last_completion_len = 0
            for engine_idx, eng in enumerate(engines):
                cfg = DIVERSE_CONFIGS[engine_idx]
                _, _, temp_cfg, top_p_cfg, steps_cfg = cfg
                n_attempts_this += 1
                total_attempts += 1
                out = eng.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    steps_per_block=steps_cfg,
                    temperature=temp_cfg,
                    top_p=top_p_cfg,
                    speculative_k=args.speculative_k,
                    speculative_threshold=args.speculative_threshold,
                    speculative_tree_k=args.speculative_tree_k,
                    speculative_band_low=args.speculative_band_low,
                    speculative_max_proposals_branch_1=args.speculative_max_proposals_branch_1,
                )
                total_forwards += out.num_forwards
                total_tokens += int(out.sequences.shape[1] - prompt_ids.shape[1])
                decoded = tok.decode(
                    out.sequences[0, prompt_ids.shape[1]:].cpu().tolist(),
                    skip_special_tokens=True,
                )
                last_decoded = decoded
                last_completion_len = int(out.sequences.shape[1] - prompt_ids.shape[1])
                if _check_completion(decoded, row):
                    passed = True
                    break
            if passed:
                n_pass += 1
            result.per_problem.append({
                "task_id": task_id,
                "passed": bool(passed),
                "seconds": time.time() - t_problem_start,
                "completion_len": last_completion_len,
                "n_attempts": n_attempts_this,
                "winning_config_idx": (n_attempts_this - 1) if passed else -1,
            })
            decoded = last_decoded
        else:
            out = engine.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                steps_per_block=args.steps_per_block,
                temperature=args.temperature,
                top_p=args.top_p,
                speculative_k=args.speculative_k,
                speculative_threshold=args.speculative_threshold,
                speculative_tree_k=args.speculative_tree_k,
                speculative_band_low=args.speculative_band_low,
                speculative_max_proposals_branch_1=args.speculative_max_proposals_branch_1,
            )
            total_forwards += out.num_forwards
            total_tokens += int(out.sequences.shape[1] - prompt_ids.shape[1])
            decoded = tok.decode(
                out.sequences[0, prompt_ids.shape[1]:].cpu().tolist(),
                skip_special_tokens=True,
            )
            passed = _check_completion(decoded, row)
            if passed:
                n_pass += 1
            result.per_problem.append({
                "task_id": task_id,
                "passed": bool(passed),
                "seconds": time.time() - t_problem_start,
                "completion_len": int(out.sequences.shape[1] - prompt_ids.shape[1]),
            })

        # Print the first 3 completions so we can debug "0% pass@1" at a glance.
        if i < 3:
            print(f"\n--- problem {i} ({task_id}) — passed={passed} ---")
            print(f"prompt[:200]: {row['prompt'][:200]!r}")
            print(f"completion[:400]: {decoded[:400]!r}")
            print(f"extracted code[:400]: {_extract_code(decoded)[:400]!r}")
            print(f"--- end problem {i} ---\n")

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            if diverse_n > 0:
                avg_attempts = total_attempts / (i + 1)
                print(f"[{i+1}/{len(items)}] pass@N = {n_pass/(i+1):.4f}  "
                      f"avg attempts = {avg_attempts:.2f}  "
                      f"avg = {elapsed/(i+1):.2f}s/problem")
            else:
                print(f"[{i+1}/{len(items)}] pass@1 = {n_pass/(i+1):.4f}  "
                      f"avg = {elapsed/(i+1):.2f}s/problem")

    wall = time.time() - t_start
    result.n_problems_run = len(items)
    if diverse_n > 0:
        # In diverse mode, the "single-shot" rate is what config 0 (the
        # primary preset) would have achieved alone. Recover it from
        # per_problem records: if winning_config_idx == 0, config 0 passed.
        n_pass_config0 = sum(
            1 for p in result.per_problem if p.get("winning_config_idx") == 0
        )
        result.pass_at_1_single_shot = n_pass_config0 / max(1, len(items))
        result.pass_at_1_best_of_n = n_pass / max(1, len(items))
    else:
        result.pass_at_1_single_shot = n_pass / max(1, len(items))
        result.pass_at_1_best_of_n = result.pass_at_1_single_shot
    result.seconds_per_problem = wall / max(1, len(items))
    result.total_forwards = total_forwards
    result.tokens_per_second = total_tokens / wall if wall > 0 else 0.0
    result.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    result.wall_seconds = wall

    _write(result, args.out)
    _print_summary(result)
    return 0


def _format_prompt(humaneval_prompt: str) -> str:
    return (
        "Complete the following Python function. Return only the full function "
        "definition in a ```python code block.\n\n"
        f"```python\n{humaneval_prompt}\n```"
    )


def _extract_code(text: str) -> str:
    """Pull the first/longest python block out of an LLM completion.

    LLMs are asked to wrap solutions in `````python ... `````;
    we extract the contents. If no block is found we fall back to the raw text
    (lets the model still pass when it forgets the wrapper).
    """
    blocks = _CODE_RE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def _check_completion(decoded: str, row) -> bool:
    """Execute ``code + row['test']`` in a subprocess; pass if ``check(entry_point)``
    runs without raising.

    Mirrors the proven pattern from
    ``diffucoder_experiments/.../bench/eval_diverse_fastdllm.py::run_test``.
    Replaces an earlier brittle path that called
    ``evalplus.evaluate.check_correctness`` with a stale signature and silently
    returned False on any exception.

    Failure modes that legitimately return False:
      - subprocess timeout (test hung)
      - ``check(...)`` raises (assertion failure → wrong answer)
      - generated code has SyntaxError → import-time crash before ``check``
      - empty generation → ``__OK__`` not in stdout

    Successful pass: stdout contains the literal sentinel ``__OK__``.
    """
    code = _extract_code(decoded)
    test_code = row.get("test", "")
    entry_point = row.get("entry_point", "")
    if not code or not test_code or not entry_point:
        return False
    full = (
        code + "\n\n" + test_code
        + f"\n\ntry:\n    check({entry_point})\n    print('__OK__')\nexcept Exception: pass\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        p = subprocess.run(
            ["python3", path], capture_output=True, text=True, timeout=10,
        )
        return "__OK__" in p.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _write(result: BenchResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))


def _print_summary(r: BenchResult) -> None:
    print("\n" + "=" * 60)
    print("Phase-1 benchmark summary")
    print("=" * 60)
    print(f"  adapter:                {r.adapter}")
    print(f"  cache / scheduler:      {r.cache} / {r.scheduler}")
    print(f"  sampler:                {r.sampler}")
    print(f"  problems run:           {r.n_problems_run}")
    print(f"  pass@1 (single-shot):   {r.pass_at_1_single_shot:.4f}")
    print(f"  pass@1 (best-of-{r.n_diverse_configs}):     {r.pass_at_1_best_of_n:.4f}")
    print(f"  s/problem:              {r.seconds_per_problem:.2f}")
    print(f"  tokens/sec:             {r.tokens_per_second:.1f}")
    print(f"  total forwards:         {r.total_forwards}")
    print(f"  peak VRAM (GB):         {r.peak_vram_gb:.2f}")
    print(f"  wall (s):               {r.wall_seconds:.1f}")


if __name__ == "__main__":
    sys.exit(main())
