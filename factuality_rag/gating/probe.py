"""
factuality_rag.gating.probe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adaptive retrieval gating via single-step logit probing.

Computes entropy and logit-gap of the model's next-token distribution
to decide whether retrieval is needed.

Example (mock-mode)::

    >>> probe = GatingProbe("mistral-7b-instruct", mock_mode=True)
    >>> probe.should_retrieve("What is Python?")
    True
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any, List, NoReturn, Optional

import numpy as np

from factuality_rag.determinism import stable_seed

logger = logging.getLogger(__name__)


# ── Standalone ECE computation ────────────────────────────────


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute equal-width top-label Expected Calibration Error (ECE).

    This metric compares each prediction's top-label confidence with an
    externally supplied binary correctness outcome.  It does not infer
    correctness from model scores and it is not, by itself, a calibration
    procedure.  The interval [0, 1] is divided into *n_bins* equal-width
    bins and the function computes::

        ECE = Σ (|B_m| / N) · |avg_conf(B_m) - avg_acc(B_m)|

    Args:
        confidences: 1-D array of predicted confidences in [0, 1].
        accuracies: 1-D array of externally established binary correctness
            outcomes (0 or 1) for the corresponding predictions.
        n_bins: Number of bins (default 15).

    Returns:
        ECE value in [0, 1].

    Example::

        >>> import numpy as np
        >>> # Perfect calibration → ECE = 0
        >>> compute_ece(np.array([0.5, 0.5]), np.array([0.0, 1.0]))
        0.0
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, numbers.Integral):
        raise TypeError("n_bins must be a positive integer")
    bin_count = int(n_bins)
    if bin_count <= 0:
        raise ValueError("n_bins must be a positive integer")

    confidence_values = np.asarray(confidences)
    correctness_values = np.asarray(accuracies)
    if confidence_values.ndim != 1:
        raise ValueError("confidences must be a one-dimensional array")
    if correctness_values.ndim != 1:
        raise ValueError("accuracies must be a one-dimensional array")
    if confidence_values.shape != correctness_values.shape:
        raise ValueError("confidences and accuracies must have the same length")
    if confidence_values.size == 0:
        raise ValueError("confidences and accuracies must be non-empty")

    if confidence_values.dtype.kind not in "iuf":
        raise TypeError("confidences must contain real numbers")
    if correctness_values.dtype.kind not in "biuf":
        raise TypeError("accuracies must contain binary numeric values")

    confidence_values = confidence_values.astype(np.float64, copy=False)
    correctness_values = correctness_values.astype(np.float64, copy=False)
    if not np.isfinite(confidence_values).all():
        raise ValueError("confidences must be finite")
    if not np.isfinite(correctness_values).all():
        raise ValueError("accuracies must be finite")
    if ((confidence_values < 0.0) | (confidence_values > 1.0)).any():
        raise ValueError("confidences must lie in [0, 1]")
    if not np.isin(correctness_values, (0.0, 1.0)).all():
        raise ValueError("accuracies must contain only 0 or 1")

    bin_boundaries = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    n = confidence_values.size

    for i in range(bin_count):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < bin_count - 1:
            mask = (confidence_values >= lo) & (confidence_values < hi)
        else:
            # Last bin includes the right boundary
            mask = (confidence_values >= lo) & (confidence_values <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        avg_conf = confidence_values[mask].mean()
        avg_acc = correctness_values[mask].mean()
        ece += (bin_size / n) * abs(avg_conf - avg_acc)

    return float(ece)


class GatingProbe:
    """Single-step logit probe for adaptive retrieval gating.

    When the model is confident (low entropy, large logit gap) the
    probe signals that retrieval can be skipped.

    Args:
        generator_model_hf: HuggingFace model identifier for the
                            generator whose logits are probed.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        temp: Softmax temperature used by the entropy calculation.  A fixed
              value is not evidence that the gate or its probabilities are
              calibrated.
        mock_mode: If ``True``, simulate logits deterministically
                   without loading the model.
        nonfinite_policy: ``"raise"`` stops an experiment when a gate
                          signal is invalid; ``"retrieve"`` is an explicit
                          conservative fallback for a serving boundary.

    Example::

        >>> probe = GatingProbe("mistral-7b-instruct", mock_mode=True)
        >>> isinstance(probe.should_retrieve("hello"), bool)
        True
    """

    def __init__(
        self,
        generator_model_hf: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cuda",
        temp: float = 1.0,
        mock_mode: bool = False,
        nonfinite_policy: str = "raise",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.generator_model_hf = generator_model_hf
        self.device = device
        self.temp = temp
        self.mock_mode = mock_mode
        if nonfinite_policy not in {"raise", "retrieve"}:
            raise ValueError("nonfinite_policy must be 'raise' or 'retrieve'")
        self.nonfinite_policy = nonfinite_policy

        # Pre-loaded or lazy-loaded model & tokenizer
        self._model = model
        self._tokenizer = tokenizer

    # ── Lazy loading ──────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load HuggingFace causal LM and tokenizer.

        Uses the shared :mod:`factuality_rag.model_registry` so
        that the same weights are reused by the generator.

        Skipped entirely in mock-mode.
        """
        if self.mock_mode or (self._model is not None and self._tokenizer is not None):
            return
        from factuality_rag.model_registry import get_model, get_tokenizer

        if self._model is None:
            logger.info("Loading generator model '%s' via registry …", self.generator_model_hf)
            self._model = get_model(self.generator_model_hf, device=self.device)
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(self.generator_model_hf)

    def _model_input_device(self) -> Any:
        """Return the loaded model's real input-embedding device."""

        if self._model is None:
            raise RuntimeError("gating model is not loaded")

        get_input_embeddings = getattr(self._model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embeddings = get_input_embeddings()
            embedding_device = getattr(getattr(embeddings, "weight", None), "device", None)
            if embedding_device is not None and str(embedding_device) != "meta":
                return embedding_device

        model_device = getattr(self._model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device

        parameters = getattr(self._model, "parameters", None)
        if callable(parameters):
            try:
                parameter_device = getattr(next(parameters()), "device", None)
            except StopIteration:
                parameter_device = None
            if parameter_device is not None and str(parameter_device) != "meta":
                return parameter_device

        raise RuntimeError("could not determine the gating model's input device")

    # ── Core gating logic ─────────────────────────────────────

    def should_retrieve(
        self,
        prompt: str,
        probe_tokens: int = 1,
        entropy_thresh: float = 1.2,
        logit_gap_thresh: float = 2.0,
    ) -> bool:
        """Decide whether retrieval is needed for *prompt*.

        Performs a single forward pass (no full decoding), computes
        entropy and logit-gap of the next-token distribution, and
        returns ``True`` (retrieve) when the model is uncertain.

        When ``probe_tokens > 1``, the model autoregressively generates
        *probe_tokens* positions and the entropy / logit-gap are
        averaged across all positions.

        Decision rule::

            retrieve = (entropy > entropy_thresh) or (logit_gap < logit_gap_thresh)

        Args:
            prompt: Input prompt string.
            probe_tokens: Number of leading logit positions to probe
                          (default 1 = next token only).
            entropy_thresh: Entropy threshold above which retrieval
                            is triggered.
            logit_gap_thresh: Minimum difference between the top-2
                              logits below which retrieval is triggered.

        Returns:
            ``True`` if retrieval should happen, ``False`` otherwise.

        Example::

            >>> probe = GatingProbe("mistral-7b-instruct", mock_mode=True)
            >>> probe.should_retrieve("What is the capital of France?")
            True
        """
        if isinstance(probe_tokens, bool) or not isinstance(probe_tokens, numbers.Integral):
            raise TypeError("probe_tokens must be a positive integer")
        probe_token_count = int(probe_tokens)
        if probe_token_count <= 0:
            raise ValueError("probe_tokens must be a positive integer")

        entropy_threshold = self._validate_finite_real("entropy_thresh", entropy_thresh)
        logit_gap_threshold = self._validate_finite_real("logit_gap_thresh", logit_gap_thresh)
        self._validate_finite_real("temperature", self.temp, positive=True)

        if probe_token_count > 1:
            # Multi-token probe: average entropy and logit-gap over k positions
            all_logits = self._get_multi_token_logits(prompt, probe_token_count)
            entropies = [self._compute_entropy(lg) for lg in all_logits]
            gaps = [self._compute_logit_gap(lg) for lg in all_logits]
            entropy = float(np.mean(entropies))
            logit_gap = float(np.mean(gaps))
        else:
            logits = self._get_next_token_logits(prompt, 1)
            entropy = self._compute_entropy(logits)
            logit_gap = self._compute_logit_gap(logits)

        if not math.isfinite(entropy) or not math.isfinite(logit_gap):
            message = f"Non-finite gating signal: entropy={entropy!r}, logit_gap={logit_gap!r}"
            if self.nonfinite_policy == "retrieve":
                logger.error("%s; forcing retrieval.", message)
                return True
            raise FloatingPointError(message)

        should = entropy > entropy_threshold or logit_gap < logit_gap_threshold
        logger.debug(
            "Gating: entropy=%.4f (thresh=%.2f), gap=%.4f (thresh=%.2f) → %s",
            entropy,
            entropy_threshold,
            logit_gap,
            logit_gap_threshold,
            "RETRIEVE" if should else "SKIP",
        )
        return should

    @staticmethod
    def _validate_finite_real(name: str, value: Any, *, positive: bool = False) -> float:
        """Validate a runtime numeric configuration value without coercing strings."""
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite")
        if positive and numeric_value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
        return numeric_value

    # ── Calibration ───────────────────────────────────────────

    def calibrate_temperature(
        self,
        dev_prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> NoReturn:
        """Reject the former label-free temperature-calibration path.

        A scientifically meaningful fit requires an explicit prediction
        target, a frozen correctness rule, and a disjoint calibration split.
        The former implementation ignored *targets* and manufactured labels
        from the same logit gap it claimed to calibrate.  Returning a
        temperature from those inputs would therefore be misleading.

        Args:
            dev_prompts: Reserved for a future supervised implementation.
            targets: Reserved for explicit, provenance-bound targets in a
                future supervised implementation.

        Returns:
            This method does not return.  It raises ``NotImplementedError``
            without probing a model or changing ``self.temp``.

        Raises:
            NotImplementedError: Always, until a label-backed calibration
                protocol is implemented.
        """
        raise NotImplementedError(
            "temperature calibration requires explicit provenance-bound targets, "
            "a frozen correctness rule, and a disjoint calibration split"
        )

    # ── Internal helpers ──────────────────────────────────────

    def _get_next_token_logits(self, prompt: str, n: int = 1) -> np.ndarray:
        """Forward pass → return logits for the next token position.

        Args:
            prompt: Input text.
            n: Ignored (kept for API compat). Use
               :meth:`_get_multi_token_logits` for multi-position probing.

        Returns:
            1-D numpy array of logits (vocab-sized) for position -1.
        """
        if self.mock_mode:
            rng = np.random.RandomState(stable_seed("gating.mock_logits", prompt, 0))
            # Simulate a vocab of size 32000 (Llama-like)
            mock_logits: np.ndarray = rng.randn(32000).astype(np.float32)
            return mock_logits

        self._load_model()
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("gating model and tokenizer must both be loaded")
        input_device = self._model_input_device()
        tokens = self._tokenizer(prompt, return_tensors="pt").to(input_device)
        with torch.no_grad():
            outputs = self._model(**tokens)
        # outputs.logits shape: (1, seq_len, vocab_size)
        logits: np.ndarray = outputs.logits[0, -1, :].cpu().numpy()
        return logits

    def _get_multi_token_logits(self, prompt: str, k: int = 3) -> List[np.ndarray]:
        """Autoregressive forward pass → logits for *k* token positions.

        Generates *k* tokens one-by-one (greedy), collecting logits
        at each step.  In mock-mode uses seeded RNG per step.

        Args:
            prompt: Input text.
            k: Number of successive token positions to probe.

        Returns:
            List of *k* 1-D numpy logit arrays (vocab-sized).

        Example::

            >>> probe = GatingProbe("x", mock_mode=True)
            >>> logits_list = probe._get_multi_token_logits("hello", k=3)
            >>> len(logits_list)
            3
        """
        if self.mock_mode:
            results: List[np.ndarray] = []
            for step in range(k):
                seed = stable_seed("gating.mock_logits", prompt, step)
                rng = np.random.RandomState(seed)
                results.append(rng.randn(32000).astype(np.float32))
            return results

        self._load_model()
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("gating model and tokenizer must both be loaded")
        input_device = self._model_input_device()
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(input_device)
        results_real: List[np.ndarray] = []

        for _ in range(k):
            with torch.no_grad():
                outputs = self._model(input_ids)
            logits = outputs.logits[0, -1, :]
            results_real.append(logits.cpu().numpy())
            # Greedily append the top token for next step
            next_token = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return results_real

    def _compute_entropy(self, logits: np.ndarray) -> float:
        """Compute Shannon entropy of the softmax distribution.

        Args:
            logits: Raw logit array.

        Returns:
            Entropy in nats.

        Example::

            >>> probe = GatingProbe("x", mock_mode=True)
            >>> e = probe._compute_entropy(np.array([1.0, 2.0, 3.0]))
            >>> e > 0
            True
        """
        # Quantized models can expose float16 logits.  Promote before
        # softmax so the probability floor remains representable and
        # zero-probability terms cannot turn the entropy into NaN.
        scaled = np.asarray(logits, dtype=np.float32) / max(self.temp, 1e-8)
        shifted = scaled - scaled.max()
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum()
        probs = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def _compute_logit_gap(self, logits: np.ndarray) -> float:
        """Compute gap between the top-2 logit values.

        Args:
            logits: Raw logit array.

        Returns:
            Absolute difference between rank-0 and rank-1 logits.

        Example::

            >>> probe = GatingProbe("x", mock_mode=True)
            >>> probe._compute_logit_gap(np.array([5.0, 2.0, 1.0]))
            3.0
        """
        if len(logits) < 2:
            return float("inf")
        stable_logits = np.asarray(logits, dtype=np.float32)
        top2 = np.partition(stable_logits, -2)[-2:]
        return float(abs(top2[1] - top2[0]))
