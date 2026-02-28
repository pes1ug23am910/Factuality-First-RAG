"""
factuality_rag.scorer.passage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passage-level factuality scorer: NLI entailment + token/char
overlap + retriever score fusion.

Fusion formula::

    final_score = w_nli * P(entailment) + w_overlap * overlap + w_ret * ret_norm

Example (mock-mode)::

    >>> scorer = PassageScorer("mock-nli", mock_mode=True)
    >>> passages = [{"id":"1","text":"Paris is in France","combined_score":0.8}]
    >>> scored = scorer.score_passages("capital of France", passages)
    >>> "final_score" in scored[0]
    True
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PassageScorer:
    """Passage-level factuality scorer.

    Args:
        nli_model_hf: HuggingFace NLI model identifier.
        overlap_metric: ``"token"`` or ``"char"`` overlap.
        device: Torch device string.
        mock_mode: If ``True``, simulate NLI scores deterministically.
        w_nli: Weight for NLI entailment probability.
        w_overlap: Weight for token/char overlap score.
        w_ret: Weight for normalised retriever score.

    Example::

        >>> s = PassageScorer("mock", mock_mode=True)
        >>> s.w_nli + s.w_overlap + s.w_ret  # doctest: +ELLIPSIS
        1.0...
    """

    def __init__(
        self,
        nli_model_hf: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        overlap_metric: str = "token",
        device: str = "cpu",
        mock_mode: bool = False,
        w_nli: float = 0.5,
        w_overlap: float = 0.2,
        w_ret: float = 0.3,
    ) -> None:
        self.nli_model_hf = nli_model_hf
        self.overlap_metric = overlap_metric
        self.device = device
        self.mock_mode = mock_mode
        self.w_nli = w_nli
        self.w_overlap = w_overlap
        self.w_ret = w_ret

        # Lazy-loaded NLI pipeline
        self._nli_pipeline: Any = None

    # ── Lazy loading ──────────────────────────────────────────

    def _load_nli(self) -> None:
        """Lazy-load the HuggingFace NLI pipeline.

        Skipped in mock-mode.
        """
        if self._nli_pipeline is not None or self.mock_mode:
            return
        from transformers import pipeline  # type: ignore[import-untyped]

        logger.info("Loading NLI model '%s' ...", self.nli_model_hf)
        self._nli_pipeline = pipeline(
            "text-classification",
            model=self.nli_model_hf,
            device=self.device if self.device != "cpu" else -1,
        )

    # ── Public API ────────────────────────────────────────────

    def score_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score each passage for factuality and add keys in-place.

        Adds ``nli_score``, ``overlap_score``, and ``final_score``
        to each passage dict.

        Args:
            query: The user query (used as premise for NLI and for
                   overlap computation).
            passages: List of passage dicts; each should have at least
                      ``text`` and ``combined_score`` (from retriever).

        Returns:
            The same list with added score keys.

        Example::

            >>> s = PassageScorer("mock", mock_mode=True)
            >>> ps = [{"id":"0","text":"hello world","combined_score":0.7}]
            >>> out = s.score_passages("hello", ps)
            >>> 0 <= out[0]["final_score"] <= 1
            True
        """
        self._load_nli()

        # Normalise retriever scores across this passage set
        ret_scores = [p.get("combined_score", 0.0) for p in passages]
        ret_min, ret_max = (min(ret_scores), max(ret_scores)) if ret_scores else (0, 1)

        for p in passages:
            # NLI: passage is premise (evidence), query is hypothesis (claim)
            nli = self._nli_entailment(
                premise=p.get("text", ""), hypothesis=query
            )
            overlap = self._overlap(query, p.get("text", ""))
            ret_raw = p.get("combined_score", 0.0)
            ret_norm = (
                (ret_raw - ret_min) / (ret_max - ret_min)
                if ret_max > ret_min
                else 0.5
            )

            p["nli_score"] = nli
            p["overlap_score"] = overlap
            p["final_score"] = float(
                self.w_nli * nli + self.w_overlap * overlap + self.w_ret * ret_norm
            )

        return passages

    # ── NLI helper ────────────────────────────────────────────

    def _nli_entailment(self, premise: str, hypothesis: str) -> float:
        """Return P(entailment | premise, hypothesis).

        In a RAG context the **passage** is the premise (evidence) and
        the **query/claim** is the hypothesis being verified.

        Args:
            premise: The passage text (evidence).
            hypothesis: The query or claim to verify.

        Returns:
            Probability in [0, 1].

        Example::

            >>> s = PassageScorer("mock", mock_mode=True)
            >>> 0 <= s._nli_entailment("a", "b") <= 1
            True
        """
        if self.mock_mode:
            rng = np.random.RandomState(abs(hash(premise + hypothesis)) % (2**31))
            return float(rng.uniform(0.3, 0.95))

        result = self._nli_pipeline(
            f"{premise} </s></s> {hypothesis}",
            top_k=None,
        )
        # Find 'entailment' label score
        for item in result:
            if "entail" in item["label"].lower():
                return float(item["score"])
        return 0.0

    # ── Overlap helper ────────────────────────────────────────

    def _overlap(self, query: str, passage: str) -> float:
        """Compute token or character overlap score.

        Args:
            query: Query string.
            passage: Passage string.

        Returns:
            Overlap score in [0, 1].

        Example::

            >>> s = PassageScorer("mock", mock_mode=True)
            >>> s._overlap("hello world", "hello there world")
            0.666...
        """
        if self.overlap_metric == "char":
            return self._char_overlap(query, passage)
        return self._token_overlap(query, passage)

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        """F1-style token overlap.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Token-level F1 overlap in [0, 1].

        Example::

            >>> PassageScorer._token_overlap("a b c", "b c d")
            0.666...
        """
        ta = Counter(a.lower().split())
        tb = Counter(b.lower().split())
        common = sum((ta & tb).values())
        if common == 0:
            return 0.0
        precision = common / max(sum(tb.values()), 1)
        recall = common / max(sum(ta.values()), 1)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        """Character-level Jaccard overlap.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Jaccard coefficient in [0, 1].

        Example::

            >>> PassageScorer._char_overlap("abc", "bcd")
            0.5
        """
        sa, sb = set(a.lower()), set(b.lower())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
