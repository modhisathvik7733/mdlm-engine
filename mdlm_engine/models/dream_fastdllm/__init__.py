"""Dream-Coder loader with fast_dllm-patched modeling baked in.

The upstream HF Hub `Dream-org/Dream-Coder-v0-Instruct-7B` ships a stock
HF forward signature: it accepts ``past_key_values`` and ``use_cache`` but
NOT ``dual_cache`` / ``replace_position``. Without those extensions the
DreamAdapter falls back to PATH C (no caching) — see
``adapters/dream.py:_DreamCaps``.

This module ships the fast_dllm-patched ``modeling_dream.py`` (1027 LOC,
verbatim from ``Dream-Coder/instruct/src/inference/fast_dllm/`` with no
local edits) and exposes a single helper that:

  1. Snapshot-downloads the HF Hub model into the local HF cache
  2. Overlays our patched ``modeling_dream.py`` on top of the cached copy
  3. Returns ``AutoModel.from_pretrained(cache_dir, trust_remote_code=True)``

After this load, ``model.forward`` accepts ``dual_cache`` and
``replace_position``. The DreamAdapter's capability detection picks them
up automatically and routes to PATH A — the ~2× speedup path.

Usage:
    from mdlm_engine.models.dream_fastdllm import load_dream_fastdllm
    model = load_dream_fastdllm(torch_dtype=torch.bfloat16).to("cuda").eval()
    # Now `model.forward` has the fast_dllm extensions.
"""
from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_MODEL = "Dream-org/Dream-Coder-v0-Instruct-7B"


def load_dream_fastdllm(model_name: str = DEFAULT_MODEL, **from_pretrained_kwargs):
    """Load a Dream-Coder model with fast_dllm extensions wired in.

    Args:
        model_name: HF Hub model id; defaults to the official Dream-Coder.
        **from_pretrained_kwargs: forwarded to ``AutoModel.from_pretrained``.
            ``trust_remote_code=True`` is enforced.

    Returns:
        The loaded ``AutoModel`` instance with patched ``forward``.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    # 1. Get the model files (weights + stock modeling code) into the HF cache.
    cache_dir = Path(snapshot_download(model_name))

    # 2. Overlay our patched modeling_dream.py. The HF Hub repo also has a
    #    file by that name (the stock one); we replace it. snapshot_download
    #    re-checks file integrity by hash, but since trust_remote_code reads
    #    the file at load time (not at download time), our overwrite wins.
    src = Path(__file__).parent / "modeling_dream.py"
    dst = cache_dir / "modeling_dream.py"
    shutil.copy(src, dst)

    # 3. Load. AutoModel resolves modeling_dream via trust_remote_code and
    #    sees our patched file — model.forward now accepts dual_cache +
    #    replace_position.
    from_pretrained_kwargs["trust_remote_code"] = True
    return AutoModel.from_pretrained(str(cache_dir), **from_pretrained_kwargs)


__all__ = ["load_dream_fastdllm", "DEFAULT_MODEL"]
