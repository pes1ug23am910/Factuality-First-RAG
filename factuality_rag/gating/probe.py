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
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Standalone ECE computation ────────────────────────────────


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute binned Expected Calibration Error (ECE).

    Divides the confidence interval [0, 1] into *n_bins* equal-width
    bins and computes::

        ECE = Σ (|B_m| / N) · |avg_conf(B_m) - avg_acc(B_m)|

    Args:
        confidences: 1-D array of predicted confidences in [0, 1].
        accuracies: 1-D array of binary accuracy labels (0 or 1).
        n_bins: Number of bins (default 15).

    Returns:
        ECE value in [0, 1].

    Example::

        >>> import numpy as np
        >>> # Perfect calibration → ECE = 0
        >>> compute_ece(np.array([0.5, 0.5]), np.array([0.0, 1.0]))
        0.0
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            # Last bin includes the right boundary
            mask = (confidences >= lo) & (confidences <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
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
        temp: Softmax temperature for calibrated probabilities.
        mock_mode: If ``True``, simulate logits deterministically
                   without loading the model.

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
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.generator_model_hf = generator_model_hf
        self.device = device
        self.temp = temp
        self.mock_mode = mock_mode

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
        if self._model is not None or self.mock_mode:
            return
        from factuality_rag.model_registry import get_model, get_tokenizer

        logger.info("Loading generator model '%s' via registry …", self.generator_model_hf)
        self._model = get_model(self.generator_model_hf, device=self.device)
        self._tokenizer = get_tokenizer(self.generator_model_hf)

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
        if probe_tokens > 1:
            # Multi-token probe: average entropy and logit-gap over k positions
            all_logits = self._get_multi_token_logits(prompt, probe_tokens)
            entropies = [self._compute_entropy(lg) for lg in all_logits]
            gaps = [self._compute_logit_gap(lg) for lg in all_logits]
            entropy = float(np.mean(entropies))
            logit_gap = float(np.mean(gaps))
        else:
            logits = self._get_next_token_logits(prompt, 1)
            entropy = self._compute_entropy(logits)
            logit_gap = self._compute_logit_gap(logits)

        should = entropy > entropy_thresh or logit_gap < logit_gap_thresh
        logger.debug(
            "Gating: entropy=%.4f (thresh=%.2f), gap=%.4f (thresh=%.2f) → %s",
            entropy,
            entropy_thresh,
            logit_gap,
            logit_gap_thresh,
            "RETRIEVE" if should else "SKIP",
        )
        return should

    # ── Calibration ───────────────────────────────────────────

    def calibrate_temperature(
        self,
        dev_prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> float:
        """Calibrate softmax temperature on a dev set.

        Uses a simple grid search over temperatures [0.5 .. 3.0]
        to minimise the expected calibration error (ECE) of the
        next-token distribution.

        Args:
            dev_prompts: List of calibration prompts.
            targets: Optional target strings (unused in current
                     implementation — reserved for future ECE calc).

        Returns:
            Best temperature value (also saved to ``self.temp``).

        Example::

            >>> probe = GatingProbe("mistral-7b-instruct", mock_mode=True)
            >>> t = probe.calibrate_temperature(["hello", "world"])
            >>> 0.1 <= t <= 5.0
            True
        """
        if self.mock_mode:
            # Deterministic mock calibration
            self.temp = 1.0
            logger.info("Mock calibration → temp=%.2f", self.temp)
            return self.temp

        # TODO: implement proper ECE-based calibration with real logits
        best_temp = 1.0
        best_ece = float("inf")
        for t in np.arange(0.5, 3.05, 0.25):
            self.temp = float(t)
            ece = self._estimate_ece(dev_prompts)
            if ece < best_ece:
                best_ece = ece
                best_temp = float(t)

        self.temp = best_temp
        logger.info("Calibrated temperature → %.2f (ECE=%.4f)", best_temp, best_ece)
        return best_temp

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
            rng = np.random.RandomState(abs(hash(prompt)) % (2**31))
            # Simulate a vocab of size 32000 (Llama-like)
            return rng.randn(32000).astype(np.float32)

        self._load_model()
        import torch

        tokens = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**tokens)
        # outputs.logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0, -1, :].cpu().numpy()
        return logits

    def _get_multi_token_logits(
        self, prompt: str, k: int = 3
    ) -> List[np.ndarray]:
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
                seed = (abs(hash(prompt)) + step) % (2**31)
                rng = np.random.RandomState(seed)
                results.append(rng.randn(32000).astype(np.float32))
            return results

        self._load_model()
        import torch

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
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
        scaled = logits / max(self.temp, 1e-8)
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
        top2 = np.partition(logits, -2)[-2:]
        return float(abs(top2[1] - top2[0]))

    def _estimate_ece(self, prompts: List[str], n_bins: int = 15) -> float:
        """Estimate Expected Calibration Error on a set of prompts.

        Uses the maximum softmax probability as the confidence and
        computes a binned ECE.  Since we lack true labels in the
        unsupervised setting, we treat the top-1 prediction as
        "correct" when the logit gap exceeds 2.0 (a proxy for
        model certainty).

        Args:
            prompts: List of text prompts.
            n_bins: Number of confidence bins.

        Returns:
            ECE estimate.
        """
        confidences = []
        accuracies = []
        for p in prompts:
            logits = self._get_next_token_logits(p)
            scaled = logits / max(self.temp, 1e-8)
            shifted = scaled - scaled.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()
            conf = float(probs.max())
            gap = self._compute_logit_gap(logits)
            # Proxy accuracy: if the gap is large, the model is likely correct
            acc = 1.0 if gap > 2.0 else 0.0
            confidences.append(conf)
            accuracies.append(acc)
        return compute_ece(
            np.array(confidences), np.array(accuracies), n_bins=n_bins
        )
