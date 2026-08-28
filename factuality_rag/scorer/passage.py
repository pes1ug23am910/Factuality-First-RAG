"""
factuality_rag.scorer.passage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passage-level evidence/relevance scorer: query-passage NLI + token/char
overlap + retriever score fusion, with optional sentence-level
NLI and cross-encoder reranking.

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

import inspect
import logging
import math
import numbers
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from factuality_rag.determinism import stable_seed

logger = logging.getLogger(__name__)

PassageId = Union[str, int]
_NLI_MAX_LENGTH = 512
_NLI_MIN_NONEMPTY_PREMISE_TOKENS = 1


class PassageScorer:
    """Passage-level evidence/relevance scorer.

    Args:
        nli_model_hf: HuggingFace NLI model identifier.
        overlap_metric: ``"token"`` or ``"char"`` overlap.
        device: Torch device string.
        mock_mode: If ``True``, simulate NLI scores deterministically.
        w_nli: Weight for NLI entailment probability.
        w_overlap: Weight for token/char overlap score.
        w_ret: Weight for normalised retriever score.
        nli_mode: ``"passage"`` (default) or ``"sentence"`` — controls
                  whether NLI is computed on the full passage or the
                  best-matching sentence.
        nli_batch_size: Positive Transformers pipeline batch size used only
                        for full-passage NLI scoring.
        cross_encoder_model: If not ``None``, rerank passages with this
                             cross-encoder **before** NLI scoring.

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
        nli_mode: str = "passage",
        nli_batch_size: int = 8,
        cross_encoder_model: Optional[str] = None,
    ) -> None:
        if overlap_metric not in {"token", "char"}:
            raise ValueError("overlap_metric must be 'token' or 'char'")
        if nli_mode not in {"passage", "sentence"}:
            raise ValueError("nli_mode must be 'passage' or 'sentence'")
        if type(mock_mode) is not bool:
            raise TypeError("mock_mode must be exactly bool")
        if isinstance(nli_batch_size, bool) or not isinstance(nli_batch_size, int):
            raise TypeError("nli_batch_size must be a positive integer")
        if int(nli_batch_size) <= 0:
            raise ValueError("nli_batch_size must be a positive integer")
        validated_weights: List[float] = []
        for name, value in (
            ("w_nli", w_nli),
            ("w_overlap", w_overlap),
            ("w_ret", w_ret),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            validated_weights.append(numeric_value)
        if not math.isclose(sum(validated_weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("w_nli, w_overlap, and w_ret must sum to 1")

        self.nli_model_hf = nli_model_hf
        self.overlap_metric = overlap_metric
        self.device = device
        self.mock_mode = mock_mode
        self.w_nli, self.w_overlap, self.w_ret = validated_weights
        self.nli_mode = nli_mode
        self.nli_batch_size = int(nli_batch_size)
        self.cross_encoder_model = cross_encoder_model

        # Lazy-loaded NLI pipeline
        self._nli_pipeline: Any = None
        # Lazy-loaded cross-encoder
        self._cross_encoder: Any = None

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

    def _load_cross_encoder(self) -> None:
        """Lazy-load the cross-encoder reranking model.

        Skipped in mock-mode or when no cross-encoder model is set.
        """
        if self._cross_encoder is not None or self.mock_mode or not self.cross_encoder_model:
            return
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        logger.info("Loading cross-encoder '%s' ...", self.cross_encoder_model)
        self._cross_encoder = CrossEncoder(self.cross_encoder_model, device=self.device)

    # ── Public API ────────────────────────────────────────────

    def score_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score each passage for query-conditioned evidence relevance.

        Adds ``nli_score``, ``overlap_score``, and ``final_score``
        to each passage dict.  Optionally reranks via cross-encoder
        first and uses sentence-level NLI if configured.

        Args:
            query: The user query (used as the NLI hypothesis and for
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

        # ── Optional cross-encoder reranking ──────────────────
        if self.cross_encoder_model:
            self._load_cross_encoder()
            passages = self._cross_encoder_rerank(query, passages)

        # Normalise retriever scores across this passage set
        ret_scores = [p.get("combined_score", 0.0) for p in passages]
        ret_min, ret_max = (min(ret_scores), max(ret_scores)) if ret_scores else (0, 1)

        passage_nli_scores: Optional[List[float]] = None
        if self.nli_mode == "passage":
            passage_nli_scores = self._batch_nli_entailment(
                [(p.get("text", ""), query) for p in passages]
            )

        for passage_index, p in enumerate(passages):
            # NLI scoring (passage or sentence level)
            if self.nli_mode == "sentence":
                nli = self._sentence_level_nli(query=query, passage_text=p.get("text", ""))
            else:
                if passage_nli_scores is None:
                    raise RuntimeError("passage-level NLI scores were not computed")
                nli = passage_nli_scores[passage_index]

            overlap = self._overlap(query, p.get("text", ""))
            ret_raw = p.get("combined_score", 0.0)
            ret_norm = (ret_raw - ret_min) / (ret_max - ret_min) if ret_max > ret_min else 0.5

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
            rng = np.random.RandomState(stable_seed("passage_scorer.mock_nli", premise, hypothesis))
            return float(rng.uniform(0.3, 0.95))

        self._load_nli()
        if self._nli_pipeline is None:
            raise RuntimeError("NLI pipeline initialization did not produce a callable pipeline")
        max_length = self._prepare_nli_max_length([(premise, hypothesis)])
        result = self._nli_pipeline(
            {"text": premise, "text_pair": hypothesis},
            top_k=None,
            truncation="only_first",
            max_length=max_length,
        )
        return self._entailment_score_from_output(result)

    def _batch_nli_entailment(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score full-passage NLI pairs in one Transformers pipeline call.

        Mock mode deliberately delegates to the public single-pair helper so
        its stable-seed behaviour remains byte-for-byte identical. Sentence
        mode also continues to use that helper and does not call this method.
        """
        if self.mock_mode:
            return [
                self._nli_entailment(premise=premise, hypothesis=hypothesis)
                for premise, hypothesis in pairs
            ]
        if not pairs:
            return []

        self._load_nli()
        if self._nli_pipeline is None:
            raise RuntimeError("NLI pipeline initialization did not produce a callable pipeline")

        max_length = self._prepare_nli_max_length(pairs)
        result = self._nli_pipeline(
            [{"text": premise, "text_pair": hypothesis} for premise, hypothesis in pairs],
            top_k=None,
            batch_size=self.nli_batch_size,
            truncation="only_first",
            max_length=max_length,
        )
        outputs = self._normalise_nli_batch_output(result, expected_count=len(pairs))
        return [self._entailment_score_from_output(output) for output in outputs]

    def _prepare_nli_max_length(self, pairs: List[Tuple[str, str]]) -> int:
        """Return the model-aware pair limit after protecting hypothesis semantics.

        ``only_first`` deliberately permits truncation of the evidence premise,
        never the query/claim hypothesis. Before inference, a real Transformers
        tokenizer is used to prove that each hypothesis, the pair's special
        tokens, and at least one token from a non-empty premise fit intact.

        Lightweight injected test doubles may omit ``tokenizer`` entirely. In
        that case the historical application cap remains usable; production
        Transformers pipelines expose their tokenizer and take this validation
        path.
        """
        max_length = self._effective_nli_max_length()
        tokenizer = self._declared_nli_pipeline_component("tokenizer")
        if tokenizer is None:
            return max_length

        special_token_counter = getattr(tokenizer, "num_special_tokens_to_add", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(special_token_counter) or not callable(encode):
            raise RuntimeError(
                "NLI tokenizer must expose encode() and num_special_tokens_to_add() "
                "to validate hypothesis length"
            )

        try:
            pair_special_tokens = special_token_counter(pair=True)
        except Exception as exc:
            raise RuntimeError(
                "NLI tokenizer could not determine pair special-token overhead"
            ) from exc
        if (
            isinstance(pair_special_tokens, bool)
            or not isinstance(pair_special_tokens, numbers.Integral)
            or int(pair_special_tokens) < 0
        ):
            raise RuntimeError("NLI tokenizer returned an invalid pair special-token count")
        special_token_count = int(pair_special_tokens)

        hypothesis_lengths: Dict[str, int] = {}
        for premise, hypothesis in pairs:
            if hypothesis not in hypothesis_lengths:
                try:
                    hypothesis_token_ids = encode(
                        hypothesis,
                        add_special_tokens=False,
                        truncation=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "NLI tokenizer could not measure hypothesis length without truncation"
                    ) from exc
                if not isinstance(hypothesis_token_ids, (list, tuple)):
                    raise RuntimeError(
                        "NLI tokenizer returned an unsupported hypothesis token-ID shape"
                    )
                hypothesis_lengths[hypothesis] = len(hypothesis_token_ids)

            hypothesis_token_count = hypothesis_lengths[hypothesis]
            required_premise_tokens = _NLI_MIN_NONEMPTY_PREMISE_TOKENS if premise.strip() else 0
            required_length = hypothesis_token_count + special_token_count + required_premise_tokens
            if required_length > max_length:
                raise ValueError(
                    "NLI hypothesis is too long to score without truncating claim semantics: "
                    f"{hypothesis_token_count} hypothesis tokens + {special_token_count} pair "
                    f"special tokens + {required_premise_tokens} required premise tokens exceed "
                    f"the effective model limit of {max_length}"
                )
        return max_length

    def _effective_nli_max_length(self) -> int:
        """Return the strictest validated application/tokenizer/model limit."""
        candidates = [_NLI_MAX_LENGTH]
        tokenizer = self._declared_nli_pipeline_component("tokenizer")
        model = self._declared_nli_pipeline_component("model")
        config = getattr(model, "config", None)

        for source, value in (
            ("tokenizer.model_max_length", getattr(tokenizer, "model_max_length", None)),
            (
                "model.config.max_position_embeddings",
                getattr(config, "max_position_embeddings", None),
            ),
        ):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or int(value) <= 0
            ):
                raise RuntimeError(f"NLI {source} must be a positive integer")
            candidates.append(int(value))

        return min(candidates)

    def _declared_nli_pipeline_component(self, name: str) -> Any:
        """Return explicit pipeline metadata without triggering dynamic test mocks."""
        if self._nli_pipeline is None:
            return None
        try:
            inspect.getattr_static(self._nli_pipeline, name)
        except AttributeError:
            return None
        return getattr(self._nli_pipeline, name, None)

    def _entailment_score_from_output(self, result: Any) -> float:
        """Validate one pipeline result and return its entailment probability."""
        items = self._normalise_nli_output(result)
        configured_entailment_ids = self._configured_entailment_ids()
        entailment_scores: List[float] = []

        for item in items:
            raw_label = item.get("label")
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise RuntimeError("NLI pipeline returned an item without a valid label")

            label = self._normalise_nli_label(raw_label)
            is_entailment = label == "entailment"
            generic_match = re.fullmatch(r"label_(\d+)", label)
            if generic_match is not None:
                label_id = int(generic_match.group(1))
                is_entailment = label_id in configured_entailment_ids

            if not is_entailment:
                continue

            raw_score = item.get("score")
            if isinstance(raw_score, bool) or not isinstance(
                raw_score, (int, float, np.integer, np.floating)
            ):
                raise RuntimeError("NLI pipeline returned a non-numeric entailment score")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError("NLI pipeline returned an invalid entailment probability")
            entailment_scores.append(score)

        if len(entailment_scores) != 1:
            raise RuntimeError(
                "NLI output must contain exactly one unambiguous entailment label; "
                "labels such as 'not_entailment', 'non_entailment', and unresolved "
                "'LABEL_n' values are not entailment"
            )
        return entailment_scores[0]

    @staticmethod
    def _normalise_nli_batch_output(result: Any, *, expected_count: int) -> List[Any]:
        """Require exactly one independently validated output per batch input."""
        if not isinstance(result, list):
            raise RuntimeError("NLI pipeline returned an unsupported batched result type")

        # Some Transformers versions flatten a singleton batch's class-score
        # list. Preserve the single-pair normaliser's accepted shape for that
        # one unambiguous case before checking batch cardinality.
        if expected_count == 1 and result and all(isinstance(item, Mapping) for item in result):
            return [result]

        if len(result) != expected_count:
            raise RuntimeError(
                "NLI pipeline batch output must contain exactly one result per input; "
                f"expected {expected_count}, received {len(result)}"
            )
        return list(result)

    @staticmethod
    def _normalise_nli_output(result: Any) -> List[Mapping[str, Any]]:
        """Normalise supported single-input Transformers output shapes.

        Text-classification pipelines return either a mapping, a flat list of
        mappings, or (in older versions) a singleton batch containing that
        list.  Other nesting is ambiguous for this single-input call and is
        rejected rather than flattened speculatively.
        """
        if isinstance(result, Mapping):
            items: Any = [result]
        elif isinstance(result, list):
            items = result
            if len(items) == 1 and isinstance(items[0], list):
                items = items[0]
        else:
            raise RuntimeError("NLI pipeline returned an unsupported result type")

        if not items or not all(isinstance(item, Mapping) for item in items):
            raise RuntimeError("NLI pipeline returned an unsupported result shape")
        return list(items)

    @staticmethod
    def _normalise_nli_label(label: str) -> str:
        """Normalise label spelling without broad substring matching."""
        return label.strip().casefold()

    @staticmethod
    def _coerce_config_label_id(value: Any) -> Optional[int]:
        """Return a non-negative config class ID, or ``None`` if invalid."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            return int(value.strip())
        return None

    def _configured_entailment_ids(self) -> Set[int]:
        """Return class IDs explicitly and unambiguously mapped to entailment."""
        model = getattr(self._nli_pipeline, "model", None)
        config = getattr(model, "config", None)
        labels_by_id: Dict[int, Set[str]] = {}

        id2label = getattr(config, "id2label", None)
        if isinstance(id2label, Mapping):
            for raw_id, raw_label in id2label.items():
                label_id = self._coerce_config_label_id(raw_id)
                if label_id is None or not isinstance(raw_label, str):
                    continue
                label = self._normalise_nli_label(raw_label)
                if not re.fullmatch(r"label_\d+", label):
                    labels_by_id.setdefault(label_id, set()).add(label)

        label2id = getattr(config, "label2id", None)
        if isinstance(label2id, Mapping):
            for raw_label, raw_id in label2id.items():
                label_id = self._coerce_config_label_id(raw_id)
                if label_id is None or not isinstance(raw_label, str):
                    continue
                label = self._normalise_nli_label(raw_label)
                if not re.fullmatch(r"label_\d+", label):
                    labels_by_id.setdefault(label_id, set()).add(label)

        conflicting_ids = sorted(
            label_id
            for label_id, labels in labels_by_id.items()
            if "entailment" in labels and labels != {"entailment"}
        )
        if conflicting_ids:
            raise RuntimeError(
                f"NLI model config has conflicting labels for class IDs {conflicting_ids}"
            )
        entailment_ids = {
            label_id for label_id, labels in labels_by_id.items() if labels == {"entailment"}
        }
        if len(entailment_ids) > 1:
            raise RuntimeError(
                f"NLI model config defines multiple entailment class IDs: {sorted(entailment_ids)}"
            )
        return entailment_ids

    # ── Overlap helper ────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex heuristics.

        This simple punctuation/whitespace regex does not handle
        abbreviations and therefore returns heuristic sentence units.

        Args:
            text: Input text.

        Returns:
            List of sentence strings (stripped, non-empty).

        Example::

            >>> PassageScorer._split_sentences("Hello world. How are you?")
            ['Hello world', 'How are you']
        """
        if not text or not text.strip():
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [p.rstrip(".!? ").strip() for p in parts if p.strip()]
        return [s for s in sentences if len(s) > 3]

    def _sentence_level_nli(self, query: str, passage_text: str) -> float:
        """Compute sentence-level NLI: max entailment over passage sentences.

        Splits the passage into sentences and returns the maximum
        P(entailment) across all sentence-query pairs.

        Args:
            query: The user query / hypothesis.
            passage_text: The full passage text.

        Returns:
            Maximum entailment probability across sentences.

        Example::

            >>> s = PassageScorer("mock", mock_mode=True, nli_mode="sentence")
            >>> 0 <= s._sentence_level_nli("hello", "Hi there. Hello world.") <= 1
            True
        """
        sentences = self._split_sentences(passage_text)
        if not sentences:
            return self._nli_entailment(premise=passage_text, hypothesis=query)

        scores = [self._nli_entailment(premise=sent, hypothesis=query) for sent in sentences]
        return float(max(scores))

    def _cross_encoder_rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank passages using a cross-encoder relevance model.

        Returns shallow copies with a ``cross_encoder_score`` key, sorted
        descending by that score.  The caller's list and dictionaries are not
        mutated.

        Args:
            query: The user query.
            passages: List of passage dicts.

        Returns:
            Passages sorted by cross-encoder score (descending).

        Example::

            >>> s = PassageScorer("mock", mock_mode=True,
            ...     cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2")
            >>> ps = [{"id":"0","text":"a","combined_score":0.5},
            ...       {"id":"1","text":"b","combined_score":0.6}]
            >>> out = s._cross_encoder_rerank("q", ps)
            >>> all("cross_encoder_score" in p for p in out)
            True
        """
        reranked = [dict(passage) for passage in passages]
        if self.mock_mode:
            passage_ids: Set[PassageId] = set()
            for passage in reranked:
                passage_id = self._stable_passage_id(passage)
                if passage_id in passage_ids:
                    raise ValueError(f"duplicate passage id for mock cross-encoder: {passage_id!r}")
                passage_ids.add(passage_id)
                rng = np.random.RandomState(
                    stable_seed("passage_scorer.mock_cross_encoder", query, passage_id)
                )
                passage["cross_encoder_score"] = float(rng.uniform(0.1, 0.95))
            return sorted(
                reranked,
                key=lambda passage: (
                    -passage["cross_encoder_score"],
                    self._passage_id_sort_key(self._stable_passage_id(passage)),
                ),
            )

        pairs = [(query, p.get("text", "")) for p in reranked]
        scores = self._cross_encoder.predict(pairs)
        for p, score in zip(reranked, scores):
            p["cross_encoder_score"] = float(score)
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked

    @staticmethod
    def _stable_passage_id(passage: Mapping[str, Any]) -> PassageId:
        """Return the explicit stable ID required by deterministic mock reranking."""
        passage_id = passage.get("id")
        if isinstance(passage_id, bool) or not isinstance(passage_id, (str, int)):
            raise ValueError("mock cross-encoder passages require a stable string or integer 'id'")
        if isinstance(passage_id, str) and (not passage_id or passage_id != passage_id.strip()):
            raise ValueError("mock cross-encoder passage 'id' must be non-empty and trimmed")
        return passage_id

    @staticmethod
    def _passage_id_sort_key(passage_id: PassageId) -> Tuple[int, Any]:
        """Return a total-order key for deterministic score tie-breaking."""
        if isinstance(passage_id, int):
            return (0, passage_id)
        return (1, passage_id)

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
