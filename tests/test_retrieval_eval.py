"""Hand-computable tests for provenance-aware retrieval metrics."""

from __future__ import annotations

import math

import pytest

from factuality_rag.eval.retrieval import (
    compute_complete_provenance_mrr,
    compute_complete_provenance_recall,
    compute_mrr,
    compute_ndcg,
    compute_r_precision,
    evaluate_retrieval,
    evaluate_retrieval_batch,
)


def test_complete_provenance_recall_requires_every_member_by_cutoff() -> None:
    ranked = ["a", "n2", "n3", "n4", "n5", "b"]
    metrics = evaluate_retrieval(ranked, [["a", "b"]])

    assert metrics["complete_provenance_recall_at_5"] == 0.0
    assert metrics["complete_provenance_recall_at_20"] == 1.0
    assert metrics["complete_provenance_recall_at_100"] == 1.0
    assert metrics["best_alternative_group_r_precision"] == 0.5
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["complete_provenance_mrr"] == pytest.approx(1.0 / 6.0)

    expected_dcg = 1.0 + 1.0 / math.log2(7.0)  # relevant ranks 1 and 6
    expected_ideal_dcg = 1.0 + 1.0 / math.log2(3.0)
    assert metrics["best_alternative_group_ndcg_at_10"] == pytest.approx(
        expected_dcg / expected_ideal_dcg
    )


def test_one_alternative_complete_group_is_sufficient() -> None:
    # The first group is entirely absent, but the second is a complete proof.
    metrics = evaluate_retrieval(["c", "d"], [["a", "b"], ["c", "d"]])

    assert metrics == {
        "complete_provenance_recall_at_5": 1.0,
        "complete_provenance_recall_at_20": 1.0,
        "complete_provenance_recall_at_100": 1.0,
        "best_alternative_group_r_precision": 1.0,
        "mrr_at_10": 1.0,
        "complete_provenance_mrr": 0.5,
        "best_alternative_group_ndcg_at_10": 1.0,
    }


def test_documents_from_different_alternatives_do_not_form_a_complete_hit() -> None:
    groups = [["a", "b"], ["c", "d"]]

    assert compute_complete_provenance_recall(["a", "c"], groups, k=100) == 0.0
    assert compute_complete_provenance_mrr(["a", "c"], groups) == 0.0


def test_overlap_between_alternative_groups_is_valid() -> None:
    groups = [["shared", "a"], ["shared", "b"]]

    assert compute_complete_provenance_recall(["shared", "b"], groups, k=5) == 1.0


def test_r_precision_and_mrr_are_hand_computable() -> None:
    ranked = ["noise", "a", "other"]
    groups = [["a", "b"]]

    assert compute_r_precision(ranked, groups) == 0.5  # one relevant item in top R=2
    assert compute_mrr(ranked, groups) == 0.5  # first relevant item is rank 2
    assert compute_mrr(ranked, groups, k=1) == 0.0


def test_standard_and_complete_provenance_mrr_remain_distinct() -> None:
    ranked = ["noise", "a", "other", "b"]
    groups = [["a", "b"]]

    assert compute_mrr(ranked, groups, k=10) == 0.5
    assert compute_complete_provenance_mrr(ranked, groups) == 0.25
    assert compute_complete_provenance_mrr(ranked, groups, k=3) == 0.0


def test_standard_mrr_and_ndcg_are_explicitly_cut_off_at_10() -> None:
    ranked = [f"noise-{index}" for index in range(10)] + ["gold"]
    metrics = evaluate_retrieval(ranked, [["gold"]])

    assert metrics["mrr_at_10"] == 0.0
    assert metrics["best_alternative_group_ndcg_at_10"] == 0.0
    assert metrics["complete_provenance_mrr"] == pytest.approx(1.0 / 11.0)
    assert metrics["complete_provenance_recall_at_20"] == 1.0


def test_graded_ndcg_uses_exponential_gain() -> None:
    ranked = ["low", "high"]
    groups = [["low", "high"]]
    grades = {"low": 1, "high": 3}

    actual = compute_ndcg(ranked, groups, grades)
    expected_dcg = 1.0 + 7.0 / math.log2(3.0)
    expected_ideal_dcg = 7.0 + 1.0 / math.log2(3.0)
    assert actual == pytest.approx(expected_dcg / expected_ideal_dcg)


def test_empty_ranking_is_a_valid_all_zero_outcome() -> None:
    assert evaluate_retrieval([], [["gold"]]) == {
        "complete_provenance_recall_at_5": 0.0,
        "complete_provenance_recall_at_20": 0.0,
        "complete_provenance_recall_at_100": 0.0,
        "best_alternative_group_r_precision": 0.0,
        "mrr_at_10": 0.0,
        "complete_provenance_mrr": 0.0,
        "best_alternative_group_ndcg_at_10": 0.0,
    }


def test_batch_evaluator_macro_averages_unique_cases() -> None:
    cases = [
        {
            "case_id": "hit",
            "ranked_ids": ["a"],
            "gold_evidence_groups": [["a"]],
        },
        {
            "case_id": "miss",
            "ranked_ids": [],
            "gold_evidence_groups": [["b"]],
        },
    ]

    assert evaluate_retrieval_batch(cases) == {
        "n_cases": 2.0,
        "complete_provenance_recall_at_5": 0.5,
        "complete_provenance_recall_at_20": 0.5,
        "complete_provenance_recall_at_100": 0.5,
        "best_alternative_group_r_precision": 0.5,
        "mrr_at_10": 0.5,
        "complete_provenance_mrr": 0.5,
        "best_alternative_group_ndcg_at_10": 0.5,
    }


def test_batch_evaluator_rejects_duplicate_case_ids() -> None:
    cases = [
        {"case_id": "q1", "ranked_ids": ["a"], "gold_evidence_groups": [["a"]]},
        {"case_id": "q1", "ranked_ids": ["b"], "gold_evidence_groups": [["b"]]},
    ]

    with pytest.raises(ValueError, match="duplicate case_id"):
        evaluate_retrieval_batch(cases)


@pytest.mark.parametrize(
    ("cases", "error_type", "message"),
    [
        ([], ValueError, "at least one retrieval case"),
        ("not-a-batch", TypeError, "ordered sequence"),
        ([[]], TypeError, r"cases\[0\] must be a mapping"),
        (
            [{"case_id": " ", "ranked_ids": ["a"], "gold_evidence_groups": [["a"]]}],
            ValueError,
            "case_id.*must be non-empty",
        ),
        (
            [{"case_id": " q ", "ranked_ids": ["a"], "gold_evidence_groups": [["a"]]}],
            ValueError,
            "case_id.*surrounding whitespace",
        ),
        (
            [
                {
                    "case_id": "q\u0000x",
                    "ranked_ids": ["a"],
                    "gold_evidence_groups": [["a"]],
                }
            ],
            ValueError,
            "case_id.*control characters",
        ),
        (
            [{"case_id": 1, "ranked_ids": ["a"], "gold_evidence_groups": [["a"]]}],
            TypeError,
            "case_id.*must be a string",
        ),
        (
            [{"case_id": "q", "ranked_ids": ["a"]}],
            ValueError,
            "missing keys",
        ),
        (
            [
                {
                    "case_id": "q",
                    "ranked_ids": ["a"],
                    "gold_evidence_groups": [["a"]],
                    "typo": True,
                }
            ],
            ValueError,
            "unknown keys",
        ),
        (
            [
                {
                    "case_id": "q",
                    "ranked_ids": ["a", "a"],
                    "gold_evidence_groups": [["a"]],
                }
            ],
            ValueError,
            r"cases\[0\].*ranked_ids contains duplicate",
        ),
    ],
)
def test_batch_evaluator_fails_closed_on_malformed_cases(
    cases: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        evaluate_retrieval_batch(cases)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ranked", "groups", "error_type", "message"),
    [
        ("doc-a", [["doc-a"]], TypeError, "ranked_ids must be an ordered sequence"),
        (["a", "a"], [["a"]], ValueError, "ranked_ids contains duplicate"),
        ([1], [["a"]], TypeError, r"ranked_ids\[0\] must be a string"),
        ([""], [["a"]], ValueError, r"ranked_ids\[0\] must be a non-empty"),
        ([" a "], [["a"]], ValueError, r"ranked_ids\[0\].*surrounding whitespace"),
        (["a\nb"], [["a"]], ValueError, r"ranked_ids\[0\].*control characters"),
        (["a"], [], ValueError, "must contain at least one evidence group"),
        (["a"], [[]], ValueError, "must contain at least one document ID"),
        (["a"], [["  "]], ValueError, "must be a non-empty document ID"),
        (["a"], [[" a "]], ValueError, "must not contain surrounding whitespace"),
        (["a"], [{"a"}], TypeError, r"gold_evidence_groups\[0\] must be an ordered"),
        (["a"], [["a", "a"]], ValueError, "contains duplicate document ID"),
        (["a"], [["a", "b"], ["b", "a"]], ValueError, "duplicate complete groups"),
    ],
)
def test_malformed_or_duplicate_rank_and_gold_inputs_fail_closed(
    ranked: object,
    groups: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        evaluate_retrieval(ranked, groups)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("grades", "error_type", "message"),
    [
        ({"a": 1.0}, ValueError, "missing gold IDs"),
        ({"a": 1.0, "b": 1.0, "extra": 1.0}, ValueError, "non-gold IDs"),
        ({"a": True, "b": 1.0}, TypeError, "must be a positive real number"),
        ({"a": float("nan"), "b": 1.0}, ValueError, "finite and positive"),
        ({"a": 0.0, "b": 1.0}, ValueError, "finite and positive"),
        ({"a": 5e-324, "b": 1.0}, ValueError, "finite positive gain"),
        ({"a": 1024.0, "b": 1.0}, ValueError, "too large"),
    ],
)
def test_malformed_relevance_grades_fail_closed(
    grades: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        compute_ndcg(["a", "b"], [["a", "b"]], grades)  # type: ignore[arg-type]


def test_near_overflow_gains_are_normalized_before_ndcg_accumulation() -> None:
    largest_finite_grade = math.nextafter(1024.0, 0.0)
    grades = {document_id: largest_finite_grade for document_id in ("a", "b", "c")}

    score = compute_ndcg(["c", "b", "a"], [["a", "b", "c"]], grades)

    assert math.isfinite(score)
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_non_positive_cutoffs_fail_closed(bad_k: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_complete_provenance_recall(["a"], [["a"]], bad_k)


@pytest.mark.parametrize("bad_k", [True, 1.5, "5"])
def test_non_integer_cutoffs_fail_closed(bad_k: object) -> None:
    with pytest.raises(TypeError, match="must be a positive integer"):
        compute_complete_provenance_recall(["a"], [["a"]], bad_k)  # type: ignore[arg-type]
