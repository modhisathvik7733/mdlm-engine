"""ModelAdapter ABC — the single most important interface in `mdlm-engine`.

Every model-specific fact (mask token id, attention-mask convention, position-id
convention, optional logit-shift, custom cache type) lives in a `ModelAdapter`
subclass. The engine never imports anything model-specific.

Adding a new diffusion LM = subclass `ModelAdapter`, implement six methods,
`@register_adapter("model_type")`. Then run the four-test validation contract
defined in `tests/test_adapter_validation.py`.

See plan §"`ModelAdapter` ABC (final, batch-aware)" for design rationale.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from mdlm_engine.cache.base import DiffusionCache


# ---------------------------------------------------------------------------
# Output / metadata types
# ---------------------------------------------------------------------------


@dataclass
class AdapterOutput:
    """Return type of `ModelAdapter.forward()`.

    The engine sees only `logits` and `past_key_values`. The adapter is free
    to do whatever the model needs internally — apply logit-shift, convert
    DiffusionCache to/from HF Cache, handle position-id quirks — but the
    output schema is fixed.

    Attributes
    ----------
    logits :
        [B, L, V] tensor. Already shifted if the model needs it (i.e. the
        adapter has called `self.shift_logits(logits)` before returning).
    past_key_values :
        Whatever the model produced. Engine treats opaque; the cache module
        consumes via `DiffusionCache.update_from_model_output()`.
    hidden_states :
        Optional, for debugging / future speculative decoding.
    """

    logits: "torch.Tensor"
    past_key_values: Any = None
    hidden_states: "torch.Tensor | None" = None


class CacheLayout(enum.Enum):
    """How the underlying model expects past_key_values to be shaped.

    Used by `DiffusionCache` implementations to allocate the right tensor
    shapes without knowing model internals.
    """

    # Most common: tuple of (K, V) per layer, K/V shape [B, n_kv_heads, L, head_dim].
    # Dream-Coder, DiffuCoder, LLaDA all use this layout (verified).
    HF_LEGACY_KV_BLD = "hf_legacy_kv_bld"

    # transformers.cache_utils.DynamicCache (or subclass).
    HF_DYNAMIC_CACHE = "hf_dynamic_cache"

    # Adapter manages its own custom cache type entirely.
    CUSTOM = "custom"


@dataclass
class ValidationReport:
    """Output of the four-test adapter validation contract.

    See `tests/test_adapter_validation.py`. Every adapter MUST pass all four
    before being registered or shipped. Stored alongside the adapter on disk
    as `adapters/<model_type>_validation.json` for reproducibility.
    """

    adapter_name: str
    model_type: str
    logit_alignment_passed: bool
    roundtrip_equivalence_passed: bool
    cache_equivalence_passed: bool
    nan_freedom_passed: bool
    notes: list[str] = field(default_factory=list)
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return (
            self.logit_alignment_passed
            and self.roundtrip_equivalence_passed
            and self.cache_equivalence_passed
            and self.nan_freedom_passed
        )


# ---------------------------------------------------------------------------
# Registry — model_type -> adapter class
# ---------------------------------------------------------------------------


_ADAPTER_REGISTRY: dict[str, type["ModelAdapter"]] = {}


def register_adapter(model_type: str):
    """Decorator: register a `ModelAdapter` subclass for an HF `config.model_type`.

    Example
    -------
    >>> @register_adapter("dream")
    ... class DreamAdapter(ModelAdapter):
    ...     model_type = "dream"
    ...     ...
    """

    def decorator(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        if not issubclass(cls, ModelAdapter):
            raise TypeError(f"{cls.__name__} must subclass ModelAdapter")
        if model_type in _ADAPTER_REGISTRY:
            raise ValueError(
                f"adapter for model_type='{model_type}' already registered "
                f"({_ADAPTER_REGISTRY[model_type].__name__})"
            )
        cls.model_type = model_type
        _ADAPTER_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_adapter_for(model_type: str) -> type["ModelAdapter"]:
    """Lookup a registered adapter by HF `config.model_type`.

    Raises
    ------
    KeyError
        If no adapter is registered for this model_type. Error message lists
        currently-registered model types.
    """
    try:
        return _ADAPTER_REGISTRY[model_type]
    except KeyError as e:
        known = sorted(_ADAPTER_REGISTRY)
        raise KeyError(
            f"no adapter for model_type='{model_type}'. "
            f"Known: {known}. Subclass `ModelAdapter` and "
            f"`@register_adapter('{model_type}')` to add support."
        ) from e


# ---------------------------------------------------------------------------
# The ABC
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """Per-model bridge between the generic engine and a specific HF diffusion LM.

    Subclasses MUST:
        - set the class attribute `model_type` (matches HF `config.model_type`)
        - implement all `@abstractmethod` methods
        - pass the four-test validation contract before being registered

    The adapter is the ONLY place model-specific code lives. If your adapter
    grows past ~200 LOC, the abstraction is leaking — file an issue.
    """

    # Set by `@register_adapter` (or by hand on the subclass).
    model_type: str = ""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Vocabulary / special tokens
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """The single token id used as the diffusion mask placeholder.

        Examples: Dream-Coder = 151666 (`<|mask|>`).
        """

    @property
    @abstractmethod
    def pad_token_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_token_ids(self) -> list[int]:
        """Token ids that signal end-of-generation.

        For chat-formatted models this typically includes both EOS and the
        chat-template terminator (e.g. `<|im_end|>`).
        """

    # ------------------------------------------------------------------
    # Canonical input path
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_chat_template(self, messages: list[dict]) -> "torch.Tensor":
        """Convert a conversation to `[B, L]` prompt token ids on CPU.

        The engine moves the tensor to GPU. The adapter owns the chat template
        because models differ here (Dream uses `<|im_start|>...<|im_end|>`,
        LLaDA uses something else; this is one of the legitimate model-specific
        concerns).
        """

    # ------------------------------------------------------------------
    # Forward-pass conventions
    # ------------------------------------------------------------------

    @abstractmethod
    def build_position_ids(
        self,
        input_ids: "torch.Tensor",
        attention_mask_1d: "torch.Tensor | None",
    ) -> "torch.Tensor | None":
        """Construct `position_ids` for the model's forward.

        Returns None if the model wants to compute its own (some custom modeling
        code overwrites whatever you pass — reference Dream's `modeling_dream.py`
        line ~837 which always uses `arange`).

        Most diffusion LMs accept either:
            - `arange(seq_len).unsqueeze(0)` — simple
            - `cumsum(attention_mask) - 1` — matches Dream when padding is on the right
        """

    @abstractmethod
    def build_attention_mask(
        self,
        attention_mask_1d: "torch.Tensor | None",
        seq_len: int,
    ) -> "torch.Tensor | str":
        """Construct the attention mask in whatever shape the model wants.

        Returns either:
            - A 4D bool/float tensor `[B, 1, L, L]` for explicit bidirectional masks
              (Dream's convention — line ~488 in `modeling_dream.py`).
            - A 2D mask `[B, L]` for models that handle the broadcast internally.
            - The sentinel string `"bidirectional"` to indicate "no mask, attend
              to everything" (LLaDA may want this).

        Diffusion LMs are bidirectional — never apply causal masking here.
        """

    @abstractmethod
    def shift_logits(self, logits: "torch.Tensor") -> "torch.Tensor":
        """Align logits so `logits[:, i, :]` predicts `tokens[:, i]`.

        - Dream / DiffuCoder: `cat([logits[:, :1], logits[:, :-1]], dim=1)`
          (their loss is computed with logits at position i predicting token i+1,
          so we shift right to align for our use).
        - LLaDA: identity (logits already aligned to predict at the masked position).

        Mandatory test #1 of the validation contract verifies this alignment
        on a known prompt at temperature 0.
        """

    # ------------------------------------------------------------------
    # The forward call itself
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | str | None",
        position_ids: "torch.Tensor | None",
        diffusion_cache: "DiffusionCache | None",
        use_cache: bool,
        *,
        block_start: int | None = None,
        block_end: int | None = None,
        is_init: bool = False,
    ) -> AdapterOutput:
        """One forward pass through the model.

        The adapter is responsible for:
            1. Converting `diffusion_cache` to whatever the model wants
               (raw tuple-of-tensors, HF Cache, custom — adapter's choice).
            2. Calling `self.model(...)`.
            3. Calling `self.shift_logits(out.logits)` before returning.
            4. Returning logits + the model's `past_key_values` for the cache
               module to consume.

        Args ``block_start`` / ``block_end`` / ``is_init`` are advisory hints
        used by adapters that implement fast_dllm-style block-only iter
        forwards (DreamAdapter PATH A). Adapters that pass full sequence
        every step (LLaDA, DreamAdapter PATH C) ignore them. Returned logits
        always have shape ``[B, L_full, V]`` regardless — adapters that
        compute on a slice pad the rest with zeros.

        The engine never sees the model directly.
        """

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------

    def cache_layout(self) -> CacheLayout:
        """Tell DiffusionCache implementations how to allocate K/V tensors.

        Default: HF legacy tuple-of-tuples with K/V shape [B, n_kv, L, d_h].
        Override only if the model uses a custom cache type.
        """
        return CacheLayout.HF_LEGACY_KV_BLD

    @property
    def n_layers(self) -> int:
        """Number of transformer layers (for cache allocation)."""
        return int(getattr(self.model.config, "num_hidden_layers", 0))

    @property
    def n_kv_heads(self) -> int:
        """Number of K/V heads (= num_attention_heads if no GQA)."""
        cfg = self.model.config
        return int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))

    @property
    def head_dim(self) -> int:
        cfg = self.model.config
        return int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    # ------------------------------------------------------------------
    # Validation hook
    # ------------------------------------------------------------------

    def validate(
        self,
        sample_prompts: list[str] | None = None,
    ) -> ValidationReport:
        """Run the four-test validation contract.

        Default impl delegates to `tests.test_adapter_validation` so the same
        logic runs from pytest and from runtime calls. See the plan
        §"Mandatory adapter validation" for the four tests.

        If you override this you MUST still ensure all four tests pass — the
        contract is the engine's correctness guarantee.
        """
        from mdlm_engine.bench.equivalence import run_adapter_validation

        return run_adapter_validation(self, sample_prompts=sample_prompts)
