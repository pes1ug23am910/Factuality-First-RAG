"""Regression tests for batched provenance NLI evaluation."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from factuality_rag.eval.metrics import compute_nli_claim_support
from factuality_rag.pipeline.orchestrator import _build_provenance


def test_claim_support_batches_pairs_in_stable_claim_major_order() -> None:
    pairs_seen: List[List[Tuple[str, str]]] = []

    def reject_single(premise: str, claim: str) -> float:
        raise AssertionError(f"unexpected single NLI call: {premise!r}, {claim!r}")

    def batch(pairs: List[Tuple[str, str]]) -> List[float]:
        pairs_seen.append(pairs)
        return [0.1, 0.8, 0.9, 0.2]

    result = compute_nli_claim_support(
        "Claim one. Claim two.",
        [{"id": "p0", "text": "first"}, {"id": "p1", "text": "second"}],
        nli_fn=reject_single,
        nli_batch_fn=batch,
        entailment_threshold=0.7,
    )

    assert pairs_seen == [
        [
            ("first", "Claim one"),
            ("second", "Claim one"),
            ("first", "Claim two"),
            ("second", "Claim two"),
        ]
    ]
    assert result["n_supported"] == 2
    assert [detail["best_passage_id"] for detail in result["details"]] == ["p1", "p0"]


def test_claim_support_batch_matches_single_call_semantics() -> None:
    score_by_pair = {
        ("first", "Claim one"): 0.1,
        ("second", "Claim one"): 0.8,
        ("first", "Claim two"): 0.9,
        ("second", "Claim two"): 0.2,
    }
    passages = [{"id": "p0", "text": "first"}, {"id": "p1", "text": "second"}]
    single = compute_nli_claim_support(
        "Claim one. Claim two.",
        passages,
        nli_fn=lambda premise, claim: score_by_pair[(premise, claim)],
    )
    batched = compute_nli_claim_support(
        "Claim one. Claim two.",
        passages,
        nli_fn=lambda premise, claim: score_by_pair[(premise, claim)],
        nli_batch_fn=lambda pairs: [score_by_pair[pair] for pair in pairs],
    )

    assert batched == single


@pytest.mark.parametrize(
    ("scores", "error", "match"),
    [
        (0.5, TypeError, "ordered sequence"),
        ([0.5], ValueError, "exactly one probability"),
        ([True, 0.5], TypeError, "numeric entailment"),
        ([float("nan"), 0.5], ValueError, "finite probabilities"),
    ],
)
def test_claim_support_rejects_invalid_batch_results(
    scores: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        compute_nli_claim_support(
            "One claim.",
            [{"id": "p0", "text": "first"}, {"id": "p1", "text": "second"}],
            nli_fn=lambda premise, claim: 0.5,
            nli_batch_fn=lambda pairs: scores,  # type: ignore[arg-type,return-value]
        )


def test_build_provenance_prefers_batch_api_but_keeps_single_fallback() -> None:
    class BatchScorer:
        pairs: List[Tuple[str, str]] = []

        @staticmethod
        def _nli_entailment(premise: str, claim: str) -> float:
            raise AssertionError("single-pair scorer should not be called")

        @classmethod
        def _batch_nli_entailment(cls, pairs: List[Tuple[str, str]]) -> List[float]:
            cls.pairs = pairs
            return [0.9 for _ in pairs]

    provenance = _build_provenance(
        "Supported claim.",
        [{"id": "p0", "text": "evidence"}],
        BatchScorer(),
    )

    assert BatchScorer.pairs == [("evidence", "Supported claim")]
    assert provenance == {"0": ["p0"]}


def test_claim_support_does_not_call_batch_scorer_without_pairs() -> None:
    def reject_empty_batch(pairs: List[Tuple[str, str]]) -> List[float]:
        raise AssertionError(f"empty batch should not be scored: {pairs!r}")

    result = compute_nli_claim_support(
        "A supported claim.",
        [],
        nli_fn=lambda premise, claim: 0.9,
        nli_batch_fn=reject_empty_batch,
    )

    assert result == {
        "nli_claim_support": 0.0,
        "n_claims": 1,
        "n_supported": 0,
        "details": [
            {
                "claim": "A supported claim",
                "supported": False,
                "best_score": 0.0,
                "best_passage_id": None,
            }
        ],
    }


def test_build_provenance_preserves_custom_single_pair_override() -> None:
    class BaseScorer:
        @staticmethod
        def _nli_entailment(premise: str, claim: str) -> float:
            return 0.1

        @staticmethod
        def _batch_nli_entailment(pairs: List[Tuple[str, str]]) -> List[float]:
            raise AssertionError("inherited batch method bypassed custom single scorer")

    class CustomScorer(BaseScorer):
        @staticmethod
        def _nli_entailment(premise: str, claim: str) -> float:
            return 0.9

    provenance = _build_provenance(
        "Supported claim.",
        [{"id": "p0", "text": "evidence"}],
        CustomScorer(),
    )

    assert provenance == {"0": ["p0"]}
