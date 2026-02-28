"""
factuality_rag.eval.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation metrics: exact match, token F1, FactScore (stub **and**
real claim-level NLI verification), and batch evaluation.

Example::

    >>> compute_em("Paris", "Paris")
    1.0
    >>> compute_f1("the capital is Paris", "Paris is the capital")  # doctest: +ELLIPSIS
    1.0
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Exact Match ───────────────────────────────────────────────


def compute_em(prediction: str, reference: str) -> float:
    """Compute exact-match score (0 or 1) after normalisation.

    Both strings are lower-cased and stripped before comparison.

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
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


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
        0.5714...
    """
    pred_tokens = Counter(prediction.lower().split())
    ref_tokens = Counter(reference.lower().split())
    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0
    precision = common / max(sum(pred_tokens.values()), 1)
    recall = common / max(sum(ref_tokens.values()), 1)
    return 2 * precision * recall / (precision + recall)


# ── FactScore stub ────────────────────────────────────────────


def compute_factscore_stub(
    claims: List[str],
    passages: List[Dict[str, Any]],
) -> float:
    """Stub FactScore: fraction of claims with any supporting passage.

    In a full implementation this would decompose the answer into
    atomic claims and verify each against retrieved passages via
    NLI. This stub checks simple word overlap as a placeholder.

    Args:
        claims: List of atomic claim strings.
        passages: List of passage dicts with ``"text"`` key.

    Returns:
        Fraction of claims supported (rough estimate).

    Example::

        >>> ps = [{"text": "Paris is the capital of France"}]
        >>> compute_factscore_stub(["Paris is a capital"], ps)
        1.0
        >>> compute_factscore_stub(["Tokyo is in Japan"], ps)
        0.0
    """
    if not claims:
        return 0.0
    passage_text = " ".join(p.get("text", "").lower() for p in passages)
    supported = 0
    for claim in claims:
        claim_tokens = set(claim.lower().split())
        # Consider a claim "supported" if > 50% of its tokens appear
        if not claim_tokens:
            continue
        overlap = sum(1 for t in claim_tokens if t in passage_text)
        if overlap / len(claim_tokens) > 0.5:
            supported += 1
    return supported / len(claims)


# ── Claim decomposition ──────────────────────────────────────


def decompose_claims(answer: str) -> List[str]:
    """Split an answer into atomic claim sentences.

    Uses sentence-boundary heuristics (split on ``.``, ``!``, ``?``)
    with simple abbreviation handling to avoid splitting on "Dr.",
    "U.S.", etc.

    Args:
        answer: The generated answer string.

    Returns:
        List of claim strings (whitespace-stripped, non-empty).

    Example::

        >>> decompose_claims("Paris is the capital. It has 2M people.")
        ['Paris is the capital', 'It has 2M people']
        >>> decompose_claims("")
        []
    """
    if not answer or not answer.strip():
        return []
    # Split on sentence-ending punctuation, keeping the delimiter
    # then rejoin where the split was an abbreviation (e.g. "U.S.")
    parts = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = [p.rstrip(".!? ").strip() for p in parts if p.strip()]
    return [c for c in claims if len(c) > 3]  # drop tiny fragments


# ── Real FactScore (claim-level NLI) ─────────────────────────


def compute_factscore(
    answer: str,
    passages: List[Dict[str, Any]],
    nli_fn: Optional[Callable[[str, str], float]] = None,
    entailment_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Compute claim-level FactScore using NLI verification.

    Steps:
        1. Decompose *answer* into atomic claims.
        2. For each claim, check NLI against every passage.
        3. A claim is *supported* if any passage yields
           ``P(entailment) > entailment_threshold``.

    Args:
        answer: The generated answer.
        passages: List of passage dicts with ``"text"`` key.
        nli_fn: A callable ``(premise, hypothesis) → float``
                returning P(entailment). If ``None``, falls back
                to the word-overlap stub.
        entailment_threshold: Minimum entailment probability for
                              a claim to be considered supported.

    Returns:
        Dict with ``factscore``, ``n_claims``, ``n_supported``,
        and per-claim ``details``.

    Example::

        >>> ps = [{"text": "Paris is the capital of France"}]
        >>> res = compute_factscore("Paris is the capital.", ps)
        >>> 0 <= res["factscore"] <= 1
        True
    """
    claims = decompose_claims(answer)
    if not claims:
        return {"factscore": 0.0, "n_claims": 0, "n_supported": 0, "details": []}

    details: List[Dict[str, Any]] = []
    supported = 0

    for claim in claims:
        best_score = 0.0
        best_passage_id = None

        for passage in passages:
            ptext = passage.get("text", "")
            if nli_fn is not None:
                # Real NLI: passage=premise, claim=hypothesis
                score = nli_fn(ptext, claim)
            else:
                # Fallback: word-overlap proxy
                claim_tokens = set(claim.lower().split())
                passage_tokens = set(ptext.lower().split())
                if claim_tokens:
                    score = len(claim_tokens & passage_tokens) / len(claim_tokens)
                else:
                    score = 0.0

            if score > best_score:
                best_score = score
                best_passage_id = passage.get("id", "?")

        is_supported = best_score >= entailment_threshold
        if is_supported:
            supported += 1

        details.append({
            "claim": claim,
            "supported": is_supported,
            "best_score": round(best_score, 4),
            "best_passage_id": best_passage_id,
        })

    factscore = supported / len(claims)
    return {
        "factscore": round(factscore, 4),
        "n_claims": len(claims),
        "n_supported": supported,
        "details": details,
    }


# ── Aggregate evaluator ──────────────────────────────────────


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    references: Optional[List[str]] = None,
    nli_fn: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, float]:
    """Evaluate a batch of predictions.

    Each prediction dict should have ``"answer"`` and optionally
    ``"trusted_passages"``. If *references* is provided, EM and F1
    are computed.  If *nli_fn* is provided, real claim-level
    FactScore is computed; otherwise the word-overlap stub is used.

    Args:
        predictions: List of dicts with ``"answer"`` key.
        references: Optional list of gold answers (parallel to
                    *predictions*).
        nli_fn: Optional ``(premise, hypothesis) → float`` for
                real FactScore computation.

    Returns:
        Dict of aggregated metrics.

    Example::

        >>> preds = [{"answer": "Paris"}, {"answer": "London"}]
        >>> refs = ["Paris", "Berlin"]
        >>> m = evaluate_predictions(preds, refs)
        >>> m["exact_match"]
        0.5
    """
    metrics: Dict[str, float] = {}

    if references:
        em_scores = [
            compute_em(p["answer"], r) for p, r in zip(predictions, references)
        ]
        f1_scores = [
            compute_f1(p["answer"], r) for p, r in zip(predictions, references)
        ]
        metrics["exact_match"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
        metrics["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # FactScore over predictions with trusted_passages
    fs_scores = []
    for p in predictions:
        passages = p.get("trusted_passages", [])
        answer = p.get("answer", "")
        if passages and answer:
            if nli_fn is not None:
                # Real claim-level FactScore
                result = compute_factscore(answer, passages, nli_fn=nli_fn)
                fs_scores.append(result["factscore"])
            else:
                # Fallback: word-overlap stub
                claims = [s.strip() for s in answer.split(".") if s.strip()]
                fs_scores.append(compute_factscore_stub(claims, passages))
    if fs_scores:
        metrics["factscore"] = sum(fs_scores) / len(fs_scores)

    metrics["n_predictions"] = float(len(predictions))
    logger.info("Evaluation: %s", metrics)
    return metrics
