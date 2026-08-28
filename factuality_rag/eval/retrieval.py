"""Strict, provenance-aware retrieval evaluation.

``gold_evidence_groups`` encodes alternative complete evidence sets.  The
outer sequence is logical OR, while every document inside one group is
required (logical AND).  For example, ``[["a", "b"], ["c", "d"]]`` means
that retrieving either both ``a`` and ``b`` *or* both ``c`` and ``d`` is a
complete provenance hit.  The alternatives are deliberately never flattened
into one relevance set.

The public functions accept ordered sequences of non-empty string document
IDs.  Repeated results, repeated members inside a group, repeated groups, and
ambiguous/malformed input raise instead of being silently normalised.  A
document may occur in more than one alternative group; such overlap is valid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from numbers import Real
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Set, Tuple, cast

__all__ = [
    "compute_complete_provenance_recall",
    "compute_complete_provenance_mrr",
    "compute_mrr",
    "compute_ndcg",
    "compute_r_precision",
    "evaluate_retrieval",
    "evaluate_retrieval_batch",
]

_RECALL_CUTOFFS = (5, 20, 100)
_RANKING_CUTOFF = 10
_OUTPUT_METRIC_KEYS = (
    "complete_provenance_recall_at_5",
    "complete_provenance_recall_at_20",
    "complete_provenance_recall_at_100",
    "best_alternative_group_r_precision",
    "mrr_at_10",
    "complete_provenance_mrr",
    "best_alternative_group_ndcg_at_10",
)


def _is_ordered_sequence(value: object) -> bool:
    """Return whether *value* is a non-text ordered sequence."""
    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _validate_document_id(document_id: object, location: str) -> str:
    if not isinstance(document_id, str):
        raise TypeError(
            f"{location} must be a string document ID; got {type(document_id).__name__}"
        )
    if not document_id.strip():
        raise ValueError(f"{location} must be a non-empty document ID")
    if document_id != document_id.strip():
        raise ValueError(f"{location} must not contain surrounding whitespace")
    if any(ord(character) < 32 for character in document_id):
        raise ValueError(f"{location} must not contain control characters")
    return document_id


def _validate_cutoff(k: object, name: str = "k") -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{name} must be a positive integer; got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"{name} must be positive; got {k}")
    return k


def _validate_inputs(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    relevance_grades: Optional[Mapping[str, float]] = None,
) -> Tuple[Tuple[str, ...], Tuple[FrozenSet[str], ...], Dict[str, float]]:
    if not _is_ordered_sequence(ranked_ids):
        raise TypeError("ranked_ids must be an ordered sequence of document ID strings")

    validated_ranking = []
    first_rank: Dict[str, int] = {}
    for index, raw_document_id in enumerate(ranked_ids):
        document_id = _validate_document_id(raw_document_id, f"ranked_ids[{index}]")
        if document_id in first_rank:
            raise ValueError(
                "ranked_ids contains duplicate document ID "
                f"{document_id!r} at positions {first_rank[document_id]} and {index}"
            )
        first_rank[document_id] = index
        validated_ranking.append(document_id)

    if not _is_ordered_sequence(gold_evidence_groups):
        raise TypeError(
            "gold_evidence_groups must be an ordered sequence of complete evidence groups"
        )
    if not gold_evidence_groups:
        raise ValueError("gold_evidence_groups must contain at least one evidence group")

    validated_groups = []
    seen_groups: Dict[FrozenSet[str], int] = {}
    all_gold_ids: Set[str] = set()
    for group_index, raw_group in enumerate(gold_evidence_groups):
        if not _is_ordered_sequence(raw_group):
            raise TypeError(
                f"gold_evidence_groups[{group_index}] must be an ordered sequence "
                "of document ID strings"
            )
        if not raw_group:
            raise ValueError(
                f"gold_evidence_groups[{group_index}] must contain at least one document ID"
            )

        group_ids = set()
        first_group_position: Dict[str, int] = {}
        for member_index, raw_document_id in enumerate(raw_group):
            location = f"gold_evidence_groups[{group_index}][{member_index}]"
            document_id = _validate_document_id(raw_document_id, location)
            if document_id in first_group_position:
                raise ValueError(
                    f"gold_evidence_groups[{group_index}] contains duplicate document ID "
                    f"{document_id!r} at positions "
                    f"{first_group_position[document_id]} and {member_index}"
                )
            first_group_position[document_id] = member_index
            group_ids.add(document_id)

        frozen_group = frozenset(group_ids)
        if frozen_group in seen_groups:
            raise ValueError(
                "gold_evidence_groups contains duplicate complete groups at indices "
                f"{seen_groups[frozen_group]} and {group_index}"
            )
        seen_groups[frozen_group] = group_index
        validated_groups.append(frozen_group)
        all_gold_ids.update(frozen_group)

    if relevance_grades is None:
        validated_gains = {document_id: 1.0 for document_id in all_gold_ids}
    else:
        if not isinstance(relevance_grades, MappingABC):
            raise TypeError("relevance_grades must be a mapping from document ID to grade")

        validated_gains = {}
        for raw_document_id, raw_grade in relevance_grades.items():
            document_id = _validate_document_id(raw_document_id, "relevance_grades key")
            if isinstance(raw_grade, bool) or not isinstance(raw_grade, Real):
                raise TypeError(
                    f"relevance grade for {document_id!r} must be a positive real number; "
                    f"got {type(raw_grade).__name__}"
                )
            grade = float(raw_grade)
            if not math.isfinite(grade) or grade <= 0.0:
                raise ValueError(
                    f"relevance grade for {document_id!r} must be finite and positive; "
                    f"got {raw_grade!r}"
                )
            try:
                gain = math.pow(2.0, grade) - 1.0
            except OverflowError as exc:
                raise ValueError(
                    f"relevance grade for {document_id!r} is too large: {raw_grade!r}"
                ) from exc
            if not math.isfinite(gain):
                raise ValueError(f"relevance grade for {document_id!r} is too large: {raw_grade!r}")
            if gain <= 0.0:
                raise ValueError(
                    f"relevance grade for {document_id!r} must produce a finite positive gain; "
                    f"got {raw_grade!r}"
                )
            validated_gains[document_id] = gain

        supplied_ids = set(validated_gains)
        missing_ids = sorted(all_gold_ids - supplied_ids)
        extra_ids = sorted(supplied_ids - all_gold_ids)
        if missing_ids or extra_ids:
            details = []
            if missing_ids:
                details.append(f"missing gold IDs {missing_ids!r}")
            if extra_ids:
                details.append(f"non-gold IDs {extra_ids!r}")
            raise ValueError(
                "relevance_grades must cover exactly the gold IDs: " + "; ".join(details)
            )

    return tuple(validated_ranking), tuple(validated_groups), validated_gains


def _complete_provenance_recall(
    ranked_ids: Tuple[str, ...],
    gold_evidence_groups: Tuple[FrozenSet[str], ...],
    k: int,
) -> float:
    retrieved = frozenset(ranked_ids[:k])
    return float(any(group.issubset(retrieved) for group in gold_evidence_groups))


def _r_precision(
    ranked_ids: Tuple[str, ...],
    gold_evidence_groups: Tuple[FrozenSet[str], ...],
) -> float:
    scores = []
    for group in gold_evidence_groups:
        relevant_in_top_r = sum(document_id in group for document_id in ranked_ids[: len(group)])
        scores.append(relevant_in_top_r / len(group))
    return max(scores)


def _mrr(
    ranked_ids: Tuple[str, ...],
    gold_evidence_groups: Tuple[FrozenSet[str], ...],
    k: Optional[int],
) -> float:
    ranked_prefix = ranked_ids if k is None else ranked_ids[:k]
    scores = []
    for group in gold_evidence_groups:
        reciprocal_rank = 0.0
        for rank, document_id in enumerate(ranked_prefix, start=1):
            if document_id in group:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return max(scores)


def _complete_provenance_mrr(
    ranked_ids: Tuple[str, ...],
    gold_evidence_groups: Tuple[FrozenSet[str], ...],
    k: Optional[int],
) -> float:
    """Return reciprocal rank at the earliest complete evidence-group prefix."""
    ranked_prefix = ranked_ids if k is None else ranked_ids[:k]
    retrieved = set()
    for rank, document_id in enumerate(ranked_prefix, start=1):
        retrieved.add(document_id)
        if any(group.issubset(retrieved) for group in gold_evidence_groups):
            return 1.0 / rank
    return 0.0


def _discounted_cumulative_gain(
    ranked_ids: Tuple[str, ...],
    group: FrozenSet[str],
    relevance_gains: Mapping[str, float],
    k: Optional[int],
) -> float:
    ranked_prefix = ranked_ids if k is None else ranked_ids[:k]
    score = 0.0
    for rank, document_id in enumerate(ranked_prefix, start=1):
        if document_id in group:
            gain = relevance_gains[document_id]
            score += gain / math.log2(rank + 1.0)
    return score


def _ideal_discounted_cumulative_gain(
    group: FrozenSet[str],
    relevance_gains: Mapping[str, float],
    k: Optional[int],
) -> float:
    gains = sorted((relevance_gains[document_id] for document_id in group), reverse=True)
    if k is not None:
        gains = gains[:k]
    return sum(gain / math.log2(rank + 1.0) for rank, gain in enumerate(gains, start=1))


def _ndcg(
    ranked_ids: Tuple[str, ...],
    gold_evidence_groups: Tuple[FrozenSet[str], ...],
    relevance_gains: Mapping[str, float],
    k: Optional[int],
) -> float:
    scores = []
    for group in gold_evidence_groups:
        max_gain = max(relevance_gains[document_id] for document_id in group)
        scaled_gains = {
            document_id: relevance_gains[document_id] / max_gain for document_id in group
        }
        dcg = _discounted_cumulative_gain(ranked_ids, group, scaled_gains, k)
        ideal_dcg = _ideal_discounted_cumulative_gain(group, scaled_gains, k)
        # Every validated group has at least one strictly positive grade.
        scores.append(min(dcg / ideal_dcg, 1.0))
    return max(scores)


def compute_complete_provenance_recall(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Return 1 iff a complete alternative evidence group occurs in top-*k*.

    Partial coverage never counts, even when the retrieved documents are drawn
    from several different alternatives.
    """
    validated_k = _validate_cutoff(k)
    ranking, groups, _ = _validate_inputs(ranked_ids, gold_evidence_groups)
    return _complete_provenance_recall(ranking, groups, validated_k)


def compute_r_precision(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
) -> float:
    """Return the best alternative-group precision at that group's ``R``.

    Each alternative has its own ``R = len(group)``.  Taking the best complete
    alternative prevents other valid evidence routes from being treated as
    additional mandatory relevant documents.
    """
    ranking, groups, _ = _validate_inputs(ranked_ids, gold_evidence_groups)
    return _r_precision(ranking, groups)


def compute_mrr(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    k: Optional[int] = None,
) -> float:
    """Return reciprocal rank of the first relevant result, optionally at *k*.

    Scores are calculated independently for each acceptable evidence group and
    the best alternative is returned.  With binary relevance this is
    equivalent to the reciprocal rank of the first member of the union, but
    the explicit group calculation keeps the alternative-set contract clear.
    """
    validated_k = None if k is None else _validate_cutoff(k)
    ranking, groups, _ = _validate_inputs(ranked_ids, gold_evidence_groups)
    return _mrr(ranking, groups, validated_k)


def compute_complete_provenance_mrr(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    k: Optional[int] = None,
) -> float:
    """Return reciprocal rank of the earliest complete evidence route.

    Unlike standard first-relevant MRR, this metric remains zero until every
    member of at least one alternative gold group has appeared.  If a group
    first becomes complete at rank ``r``, its score is ``1 / r``.  The earliest
    completed alternative wins.  An optional *k* limits the inspected prefix.
    """
    validated_k = None if k is None else _validate_cutoff(k)
    ranking, groups, _ = _validate_inputs(ranked_ids, gold_evidence_groups)
    return _complete_provenance_mrr(ranking, groups, validated_k)


def compute_ndcg(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    relevance_grades: Optional[Mapping[str, float]] = None,
    k: Optional[int] = None,
) -> float:
    """Return the best alternative-group nDCG, optionally at *k*.

    Binary relevance (grade ``1``) is used when *relevance_grades* is omitted.
    When supplied, grades must be finite positive real numbers and cover
    exactly the union of gold document IDs.  Gain is ``2**grade - 1`` and the
    logarithmic discount is ``log2(rank + 1)``.
    """
    validated_k = None if k is None else _validate_cutoff(k)
    ranking, groups, grades = _validate_inputs(ranked_ids, gold_evidence_groups, relevance_grades)
    return _ndcg(ranking, groups, grades, validated_k)


def evaluate_retrieval(
    ranked_ids: Sequence[str],
    gold_evidence_groups: Sequence[Sequence[str]],
    relevance_grades: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute the required retrieval metrics for one query.

    The three complete-provenance recall values are binary per-query scores.
    ``mrr_at_10`` is standard first-relevant MRR@10; it does not imply that a
    multi-document provenance group is complete.  ``complete_provenance_mrr``
    separately reports the reciprocal rank where an entire alternative first
    becomes available. ``best_alternative_group_r_precision`` and
    ``best_alternative_group_ndcg_at_10`` are bespoke best-over-valid-route
    values, not unqualified standard R-precision or nDCG.

    An empty ranking is a valid retrieval outcome and receives zero for every
    metric.  Gold evidence must be present because the metrics are otherwise
    undefined.
    """
    ranking, groups, grades = _validate_inputs(ranked_ids, gold_evidence_groups, relevance_grades)
    return {
        "complete_provenance_recall_at_5": _complete_provenance_recall(
            ranking, groups, _RECALL_CUTOFFS[0]
        ),
        "complete_provenance_recall_at_20": _complete_provenance_recall(
            ranking, groups, _RECALL_CUTOFFS[1]
        ),
        "complete_provenance_recall_at_100": _complete_provenance_recall(
            ranking, groups, _RECALL_CUTOFFS[2]
        ),
        "best_alternative_group_r_precision": _r_precision(ranking, groups),
        "mrr_at_10": _mrr(ranking, groups, _RANKING_CUTOFF),
        "complete_provenance_mrr": _complete_provenance_mrr(ranking, groups, None),
        "best_alternative_group_ndcg_at_10": _ndcg(ranking, groups, grades, _RANKING_CUTOFF),
    }


def evaluate_retrieval_batch(
    cases: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    """Strictly validate retrieval cases and macro-average their metrics.

    Every case must be a mapping with exactly ``case_id``, ``ranked_ids``, and
    ``gold_evidence_groups``, plus optional ``relevance_grades``.  Case IDs must
    be unique non-empty strings.  Unknown keys are rejected to catch schema
    drift and misspellings rather than silently dropping data.

    Returns the arithmetic mean of every per-query metric and ``n_cases`` as a
    float, matching the existing evaluator's numeric result-dictionary style.
    An empty batch is invalid because no macro average is defined.
    """
    if not _is_ordered_sequence(cases):
        raise TypeError("cases must be an ordered sequence of retrieval-case mappings")
    if not cases:
        raise ValueError("cases must contain at least one retrieval case")

    required_keys = frozenset({"case_id", "ranked_ids", "gold_evidence_groups"})
    allowed_keys = required_keys | {"relevance_grades"}
    seen_case_ids: Dict[str, int] = {}
    totals = {metric_name: 0.0 for metric_name in _OUTPUT_METRIC_KEYS}

    for case_index, raw_case in enumerate(cases):
        if not isinstance(raw_case, MappingABC):
            raise TypeError(f"cases[{case_index}] must be a mapping; got {type(raw_case).__name__}")
        for raw_key in raw_case:
            if not isinstance(raw_key, str):
                raise TypeError(
                    f"cases[{case_index}] keys must be strings; got {type(raw_key).__name__}"
                )

        case_keys = set(raw_case)
        missing_keys = sorted(required_keys - case_keys)
        unknown_keys = sorted(case_keys - allowed_keys)
        if missing_keys or unknown_keys:
            details = []
            if missing_keys:
                details.append(f"missing keys {missing_keys!r}")
            if unknown_keys:
                details.append(f"unknown keys {unknown_keys!r}")
            raise ValueError(f"cases[{case_index}] has invalid schema: " + "; ".join(details))

        raw_case_id = raw_case["case_id"]
        if not isinstance(raw_case_id, str):
            raise TypeError(
                f"cases[{case_index}]['case_id'] must be a string; got {type(raw_case_id).__name__}"
            )
        if not raw_case_id.strip():
            raise ValueError(f"cases[{case_index}]['case_id'] must be non-empty")
        if raw_case_id != raw_case_id.strip():
            raise ValueError(
                f"cases[{case_index}]['case_id'] must not contain surrounding whitespace"
            )
        if any(ord(character) < 32 for character in raw_case_id):
            raise ValueError(f"cases[{case_index}]['case_id'] must not contain control characters")
        if raw_case_id in seen_case_ids:
            raise ValueError(
                f"cases contains duplicate case_id {raw_case_id!r} at indices "
                f"{seen_case_ids[raw_case_id]} and {case_index}"
            )
        seen_case_ids[raw_case_id] = case_index

        ranked_ids = cast(Sequence[str], raw_case["ranked_ids"])
        gold_groups = cast(Sequence[Sequence[str]], raw_case["gold_evidence_groups"])
        grades = cast(Optional[Mapping[str, float]], raw_case.get("relevance_grades"))
        try:
            metrics = evaluate_retrieval(ranked_ids, gold_groups, grades)
        except TypeError as exc:
            raise TypeError(f"cases[{case_index}] ({raw_case_id!r}): {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"cases[{case_index}] ({raw_case_id!r}): {exc}") from exc

        for metric_name in _OUTPUT_METRIC_KEYS:
            totals[metric_name] += metrics[metric_name]

    n_cases = len(cases)
    aggregated = {"n_cases": float(n_cases)}
    aggregated.update(
        {metric_name: totals[metric_name] / n_cases for metric_name in _OUTPUT_METRIC_KEYS}
    )
    return aggregated
