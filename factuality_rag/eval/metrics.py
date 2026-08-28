"""
factuality_rag.eval.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation metrics: exact match, token F1, an explicitly labelled lexical
support proxy, sentence-unit NLI evidence scoring, and batch evaluation.

Example::

    >>> compute_em("Paris", "Paris")
    1.0
    >>> compute_f1("the capital is Paris", "Paris is the capital")  # doctest: +ELLIPSIS
    1.0
"""

from __future__ import annotations

import logging
import math
import numbers
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from factuality_rag.abstention import is_canonical_abstention

Reference = Union[str, Sequence[str]]

logger = logging.getLogger(__name__)


# ── Exact Match ───────────────────────────────────────────────


def _normalize_answer(text: str) -> str:
    """Apply the conventional SQuAD/NQ-open answer normalizer."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def compute_em(prediction: str, reference: str) -> float:
    """Compute exact-match score (0 or 1) after normalisation.

    Both strings use the conventional SQuAD/NQ-open normalizer: lower-case,
    remove ASCII punctuation and English articles, then collapse whitespace.

    Args:
        prediction: Model prediction.
        reference: Gold reference.

    Returns:
        ``1.0`` if match, ``0.0`` otherwise.

    Example::

        >>> compute_em("  Paris ", "paris")
        1.0
        >>> compute_em("London", "Paris")
        0.0
    """
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def reference_aliases(reference: Reference) -> List[str]:
    """Validate and return the aliases represented by one reference item."""
    if isinstance(reference, str):
        aliases = [reference]
    elif isinstance(reference, Sequence):
        aliases = list(reference)
    else:
        raise TypeError("each reference must be a string or sequence of strings")
    if not aliases or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
        raise ValueError("each prediction requires at least one non-blank string reference")
    return aliases


def compute_em_aliases(prediction: str, reference: Reference) -> float:
    """Return the best exact-match score across all accepted aliases."""
    return max(compute_em(prediction, alias) for alias in reference_aliases(reference))


# ── Token F1 ──────────────────────────────────────────────────


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference.

    Args:
        prediction: Model prediction.
        reference: Gold reference.

    Returns:
        F1 score in [0, 1].

    Example::

        >>> compute_f1("the cat sat", "cat sat on mat")
        0.6666...
    """
    pred_tokens = Counter(_normalize_answer(prediction).split())
    ref_tokens = Counter(_normalize_answer(reference).split())
    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0
    precision = common / max(sum(pred_tokens.values()), 1)
    recall = common / max(sum(ref_tokens.values()), 1)
    return 2 * precision * recall / (precision + recall)


def compute_f1_aliases(prediction: str, reference: Reference) -> float:
    """Return the best token-F1 score across all accepted aliases."""
    return max(compute_f1(prediction, alias) for alias in reference_aliases(reference))


# ── Lexical support proxy ─────────────────────────────────────


def compute_lexical_support(
    claims: List[str],
    passages: List[Dict[str, Any]],
) -> float:
    """Return a lexical-overlap support proxy for caller-supplied text units.

    This is deliberately *not* FactScore or an NLI metric. It checks whether
    more than half of each unit's whitespace tokens occur in the combined
    passage text and returns the fraction of units that clear that heuristic.

    Args:
        claims: List of caller-supplied text-unit strings.
        passages: List of passage dicts with ``"text"`` key.

    Returns:
        Fraction of text units clearing the lexical-overlap heuristic.

    Example::

        >>> ps = [{"text": "Paris is the capital of France"}]
        >>> compute_lexical_support(["Paris is a capital"], ps)
        1.0
        >>> compute_lexical_support(["Tokyo is in Japan"], ps)
        0.0
    """
    if not claims:
        return 0.0
    passage_tokens = {
        token for passage in passages for token in str(passage.get("text", "")).lower().split()
    }
    supported = 0
    for claim in claims:
        claim_tokens = set(claim.lower().split())
        # Consider a caller-supplied unit "supported" if > 50% of its tokens appear.
        if not claim_tokens:
            continue
        overlap = sum(1 for t in claim_tokens if t in passage_tokens)
        if overlap / len(claim_tokens) > 0.5:
            supported += 1
    return supported / len(claims)


# ── Heuristic sentence-unit splitting (compatibility API) ───


def decompose_claims(answer: str) -> List[str]:
    """Split an answer into heuristic sentence-like units.

    The regex splits only when sentence-ending punctuation is followed by
    whitespace, strips terminal punctuation, and drops fragments shorter than
    four characters. It does not perform atomic-fact decomposition or special
    abbreviation handling.

    Args:
        answer: The generated answer string.

    Returns:
        List of heuristic sentence-unit strings (stripped and length-filtered).

    Example::

        >>> decompose_claims("Paris is the capital. It has 2M people.")
        ['Paris is the capital', 'It has 2M people']
        >>> decompose_claims("")
        []
    """
    if not answer or not answer.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace. This simple
    # regex has no abbreviation-aware or atomic-fact decomposition stage.
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = [p.rstrip(".!? ").strip() for p in parts if p.strip()]
    return [c for c in claims if len(c) > 3]  # drop tiny fragments


# ── Sentence-unit NLI evidence-support primitive ─────────────


def compute_nli_claim_support(
    answer: str,
    passages: List[Dict[str, Any]],
    *,
    nli_fn: Callable[[str, str], float],
    nli_batch_fn: Optional[Callable[[List[Tuple[str, str]]], Sequence[float]]] = None,
    entailment_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Compute an NLI evidence-support rate over heuristic sentence units.

    This is an internal evidence-support primitive, not the published FActScore
    methodology. It uses heuristic sentence-like units rather than atomic facts
    and therefore must never be labelled FactScore.

    Steps:
        1. Split *answer* into heuristic sentence-like units.
        2. For each unit, check NLI against every passage.
        3. A unit is *supported* if any passage yields
           ``P(entailment) >= entailment_threshold``.

    Args:
        answer: The generated answer.
        passages: List of passage dicts with ``"text"`` key.
        nli_fn: A callable ``(premise, hypothesis) → float`` returning
                P(entailment). A lexical fallback is never used.
        nli_batch_fn: Optional callable accepting all ``(premise, hypothesis)``
                      pairs in stable claim-major order. When omitted, ``nli_fn``
                      is called once per pair for backward compatibility.
        entailment_threshold: Minimum entailment probability for
                              a sentence unit to be considered supported.

    Returns:
        Dict with compatibility keys ``nli_claim_support``, ``n_claims``,
        ``n_supported``, and per-unit ``details``. Here ``claim`` means one
        regex-derived sentence unit, not an atomic fact.

    Example::

        >>> ps = [{"text": "Paris is the capital of France"}]
        >>> res = compute_nli_claim_support(
        ...     "Paris is the capital.", ps, nli_fn=lambda premise, claim: 0.9
        ... )
        >>> 0 <= res["nli_claim_support"] <= 1
        True
    """
    if not callable(nli_fn):
        raise TypeError("nli_fn must be callable; lexical fallback is not permitted")
    if nli_batch_fn is not None and not callable(nli_batch_fn):
        raise TypeError("nli_batch_fn must be callable or None")
    if isinstance(entailment_threshold, bool) or not isinstance(entailment_threshold, numbers.Real):
        raise TypeError("entailment_threshold must be numeric")
    threshold = float(entailment_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("entailment_threshold must be finite and in [0, 1]")

    claims = decompose_claims(answer)
    if not claims:
        return {
            "nli_claim_support": 0.0,
            "n_claims": 0,
            "n_supported": 0,
            "details": [],
        }

    validated_passages: List[Tuple[Dict[str, Any], str]] = []
    for passage in passages:
        ptext = passage.get("text", "")
        if not isinstance(ptext, str):
            raise TypeError("passage text must be a string")
        validated_passages.append((passage, ptext))

    batched_scores: Optional[List[float]] = None
    if nli_batch_fn is not None:
        pairs = [(ptext, claim) for claim in claims for _, ptext in validated_passages]
        # Preserve the legacy single-call semantics for an empty evidence set:
        # no scorer is invoked when there are no premise/hypothesis pairs.
        raw_scores = nli_batch_fn(pairs) if pairs else []
        if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Sequence):
            raise TypeError("nli_batch_fn must return an ordered sequence of probabilities")
        if len(raw_scores) != len(pairs):
            raise ValueError(
                "nli_batch_fn must return exactly one probability per input pair: "
                f"expected {len(pairs)}, received {len(raw_scores)}"
            )
        batched_scores = list(raw_scores)

    details: List[Dict[str, Any]] = []
    supported = 0
    pair_index = 0

    for claim in claims:
        best_score = -1.0
        best_passage_id = None

        for passage, ptext in validated_passages:
            # Real NLI: passage=premise, claim=hypothesis. Invalid scorer
            # output fails closed and is never replaced by lexical overlap.
            if batched_scores is None:
                raw_score = nli_fn(ptext, claim)
                score_source = "nli_fn"
            else:
                raw_score = batched_scores[pair_index]
                pair_index += 1
                score_source = "nli_batch_fn"
            if isinstance(raw_score, bool) or not isinstance(raw_score, numbers.Real):
                raise TypeError(f"{score_source} must return numeric entailment probabilities")
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{score_source} must return finite probabilities in [0, 1]")

            if score > best_score:
                best_score = score
                best_passage_id = passage.get("id", "?")
                if best_passage_id is None:
                    raise ValueError("passage id must not be null")

        is_supported = best_passage_id is not None and best_score >= threshold
        if is_supported:
            supported += 1

        details.append(
            {
                "claim": claim,
                "supported": is_supported,
                "best_score": round(max(best_score, 0.0), 4),
                "best_passage_id": best_passage_id,
            }
        )

    support_rate = supported / len(claims)
    return {
        "nli_claim_support": round(support_rate, 4),
        "n_claims": len(claims),
        "n_supported": supported,
        "details": details,
    }


# ── Aggregate evaluator ──────────────────────────────────────


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    references: Optional[Sequence[Reference]] = None,
    *,
    support_metric: str = "none",
    nli_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, Any]:
    """Evaluate a batch of predictions.

    Each prediction dict must have a string ``"answer"`` and may optionally
    ``"trusted_passages"``. If *references* is provided, EM and F1
    are computed. Evidence support is disabled by default. Callers may
    explicitly request the lexical proxy; publication-grade NLI aggregation
    remains unavailable until an immutable scorer identity is wired in. Blank
    answers and the generator's canonical abstention response are excluded
    from answered-only support and coverage.

    Args:
        predictions: List of dicts with ``"answer"`` key.
        references: Optional parallel list whose items are either one gold
                    answer or a non-empty sequence of gold aliases.
        support_metric: ``"none"`` or ``"lexical"``.
        nli_fn: Reserved for a future immutable NLI evaluator. Supplying it
                currently fails closed rather than inferring a metric mode.

    Returns:
        Dict of aggregated metrics.

    Example::

        >>> preds = [{"answer": "Paris"}, {"answer": "London"}]
        >>> refs = ["Paris", "Berlin"]
        >>> m = evaluate_predictions(preds, refs)
        >>> m["exact_match"]
        0.5
    """
    if support_metric not in {"none", "lexical"}:
        raise ValueError("support_metric must be 'none' or 'lexical'")
    if nli_fn is not None:
        raise ValueError(
            "batch NLI/FactScore aggregation is unavailable; an immutable scorer "
            "and atomic-fact decomposer are not wired, and the current regex "
            "splitter produces only heuristic sentence units"
        )

    metrics: Dict[str, Any] = {"support_metric": support_metric}

    answers: List[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise TypeError(f"predictions[{index}] must be a dict")
        if "answer" not in prediction:
            raise ValueError(f"predictions[{index}] must contain an answer")
        answer = prediction["answer"]
        if not isinstance(answer, str):
            raise TypeError(f"predictions[{index}].answer must be a string")
        passages = prediction.get("trusted_passages", [])
        if not isinstance(passages, list):
            raise TypeError(f"predictions[{index}].trusted_passages must be a list")
        answers.append(answer)

    if references is not None:
        if len(references) != len(predictions):
            raise ValueError(
                "references and predictions must have the same length: "
                f"{len(references)} != {len(predictions)}"
            )
        em_scores = []
        f1_scores = []
        for answer, reference in zip(answers, references):
            em_scores.append(compute_em_aliases(answer, reference))
            f1_scores.append(compute_f1_aliases(answer, reference))
        metrics["exact_match"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
        metrics["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Evidence support is computed only over non-abstaining answers. Coverage
    # and its denominator remain explicit even when support scoring is disabled.
    lexical_scores: List[float] = []
    answered_count = 0
    for p, answer in zip(predictions, answers):
        passages = p.get("trusted_passages", [])
        if answer.strip() and not is_canonical_abstention(answer):
            answered_count += 1
            if support_metric == "lexical":
                claims = [s.strip() for s in answer.split(".") if s.strip()]
                lexical_scores.append(compute_lexical_support(claims, passages))

    if support_metric == "lexical":
        lexical_count = len(lexical_scores)
        metrics["lexical_support_answered_only"] = (
            sum(lexical_scores) / lexical_count if lexical_count else 0.0
        )
        metrics["lexical_support_answered_count"] = float(lexical_count)
    metrics["answered_count"] = float(answered_count)
    metrics["answer_coverage"] = answered_count / len(predictions) if predictions else 0.0

    metrics["n_predictions"] = float(len(predictions))
    logger.info("Evaluation: %s", metrics)
    return metrics
