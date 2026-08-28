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

import importlib.util
import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── Global singletons ────────────────────────────────────────
_models: Dict[str, Any] = {}
_model_load_configs: Dict[str, tuple[str, bool, bool]] = {}
_tokenizers: Dict[str, Any] = {}
_tokenizer_load_configs: Dict[str, bool] = {}
# Each lock covers first load plus its paired object/config cache write.  Separate
# locks keep model and tokenizer initialization independent.
_model_registry_lock = threading.RLock()
_tokenizer_registry_lock = threading.RLock()


def _resolve_device_map(device: str, torch_module: Any) -> Dict[str, Any]:
    """Validate a supported load device and return an explicit device map.

    Transformers otherwise chooses the current accelerator when a quantized
    model is loaded without ``device_map``.  That silently ignores requests
    such as ``cuda:1`` and can later put gating inputs on a different device.
    """

    if not isinstance(device, str) or not device or device != device.strip():
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    try:
        requested_device = torch_module.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'") from exc

    if requested_device.type == "cpu":
        if requested_device.index is not None:
            raise ValueError("CPU device must be specified exactly as 'cpu'")
        return {"": "cpu"}
    if requested_device.type != "cuda":
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    if not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable")

    device_count = int(torch_module.cuda.device_count())
    device_index = requested_device.index
    if device_index is None:
        device_index = int(torch_module.cuda.current_device())
    if device_index < 0 or device_index >= device_count:
        raise RuntimeError(
            f"CUDA device {device!r} resolves to index {device_index}, but only "
            f"{device_count} CUDA device(s) are available"
        )
    return {"": device_index}


def _require_4bit_runtime() -> None:
    """Fail closed when the declared 4-bit runtime extra is unavailable."""

    missing = []
    for module_name in ("bitsandbytes", "accelerate"):
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "4-bit quantization was requested but required packages are missing: "
            f"{joined}. Install the 4-bit runtime with "
            "`pip install 'factuality_rag[quantization]'` "
            "or call get_model(..., quantize_4bit=False) explicitly."
        )


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
    requested_config = (device, quantize_4bit, trust_remote_code)
    with _model_registry_lock:
        if model_id in _models:
            loaded_config = _model_load_configs.get(model_id)
            if loaded_config != requested_config:
                raise RuntimeError(
                    f"model {model_id!r} is already cached with load settings "
                    f"{loaded_config!r}, which are incompatible with requested settings "
                    f"{requested_config!r}; call clear_registry() before changing precision, "
                    "device, or trust_remote_code"
                )
            return _models[model_id]

        import torch

        explicit_device_map = _resolve_device_map(device, torch)
        from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

        kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }

        if quantize_4bit:
            _require_4bit_runtime()
            from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = explicit_device_map
            logger.info("Loading '%s' in 4-bit quantisation …", model_id)
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        _models[model_id] = model
        _model_load_configs[model_id] = requested_config
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
    with _tokenizer_registry_lock:
        if model_id in _tokenizers:
            loaded_trust_remote_code = _tokenizer_load_configs.get(model_id)
            if loaded_trust_remote_code != trust_remote_code:
                raise RuntimeError(
                    f"tokenizer {model_id!r} is already cached with trust_remote_code="
                    f"{loaded_trust_remote_code!r}, incompatible with requested "
                    f"trust_remote_code={trust_remote_code!r}; call clear_registry() first"
                )
            return _tokenizers[model_id]

        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        _tokenizers[model_id] = tokenizer
        _tokenizer_load_configs[model_id] = trust_remote_code
        logger.info("Tokenizer '%s' loaded and cached.", model_id)
        return tokenizer


def clear_registry() -> None:
    """Remove all cached models and tokenizers.

    Useful in tests or when switching model configurations.

    Example::

        >>> clear_registry()  # always safe to call
    """
    with _model_registry_lock:
        _models.clear()
        _model_load_configs.clear()
    with _tokenizer_registry_lock:
        _tokenizers.clear()
        _tokenizer_load_configs.clear()
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
    with _model_registry_lock:
        return model_id in _models
