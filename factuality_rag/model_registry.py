"""
factuality_rag.model_registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Singleton registry for shared model instances.

Avoids loading the same 7B-parameter model multiple times (e.g.
once for the gating probe and once for the generator).

Usage::

    >>> from factuality_rag.model_registry import get_model, get_tokenizer
    >>> model = get_model("mistralai/Mistral-7B-Instruct-v0.3")
    >>> tokenizer = get_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Global singletons ────────────────────────────────────────
_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}


def get_model(
    model_id: str,
    device: str = "cuda",
    quantize_4bit: bool = True,
    trust_remote_code: bool = False,
) -> Any:
    """Return a cached ``AutoModelForCausalLM`` instance.

    On first call for a given *model_id* the model is loaded (with
    optional 4-bit quantisation).  Subsequent calls return the same
    object.

    Args:
        model_id: HuggingFace model identifier.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        quantize_4bit: Use ``bitsandbytes`` 4-bit quantisation.
        trust_remote_code: Passed to ``from_pretrained()``.

    Returns:
        A ``PreTrainedModel`` in eval mode.

    Example::

        >>> # Only runs when GPU + model available
        >>> # model = get_model("mistralai/Mistral-7B-Instruct-v0.3")
    """
    if model_id in _models:
        return _models[model_id]

    import torch
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }

    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Loading '%s' in 4-bit quantisation …", model_id)
        except ImportError:
            logger.warning(
                "bitsandbytes not installed – loading '%s' in full precision.",
                model_id,
            )
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = device
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    _models[model_id] = model
    logger.info("Model '%s' loaded and cached.", model_id)
    return model


def get_tokenizer(
    model_id: str,
    trust_remote_code: bool = False,
) -> Any:
    """Return a cached ``AutoTokenizer`` instance.

    Args:
        model_id: HuggingFace model identifier.
        trust_remote_code: Passed to ``from_pretrained()``.

    Returns:
        A ``PreTrainedTokenizerFast``.
    """
    if model_id in _tokenizers:
        return _tokenizers[model_id]

    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    _tokenizers[model_id] = tokenizer
    logger.info("Tokenizer '%s' loaded and cached.", model_id)
    return tokenizer


def clear_registry() -> None:
    """Remove all cached models and tokenizers.

    Useful in tests or when switching model configurations.

    Example::

        >>> clear_registry()  # always safe to call
    """
    _models.clear()
    _tokenizers.clear()
    logger.info("Model registry cleared.")


def is_loaded(model_id: str) -> bool:
    """Check whether *model_id* is already cached.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        ``True`` if the model is in the registry.

    Example::

        >>> is_loaded("nonexistent-model")
        False
    """
    return model_id in _models
