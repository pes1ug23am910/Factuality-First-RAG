"""
tests.test_eval
~~~~~~~~~~~~~~~~
Unit tests for evaluation metrics including claim decomposition
and real FactScore.
"""

from __future__ import annotations

import pytest

from factuality_rag.eval.metrics import (
    compute_em,
    compute_f1,
    compute_factscore,
    compute_factscore_stub,
    decompose_claims,
    evaluate_predictions,
)


class TestMetrics:
    def test_exact_match_positive(self) -> None:
        assert compute_em("Paris", "paris") == 1.0

    def test_exact_match_negative(self) -> None:
        assert compute_em("London", "Paris") == 0.0

    def test_f1_identical(self) -> None:
        assert compute_f1("the cat sat", "the cat sat") == 1.0

    def test_f1_partial(self) -> None:
        score = compute_f1("the cat", "the cat sat on mat")
        assert 0.0 < score < 1.0

    def test_factscore_stub_supported(self) -> None:
        ps = [{"text": "Paris is the capital of France"}]
        assert compute_factscore_stub(["Paris is a capital"], ps) == 1.0

    def test_factscore_stub_unsupported(self) -> None:
        ps = [{"text": "Paris is the capital of France"}]
        assert compute_factscore_stub(["Tokyo is in Japan"], ps) == 0.0

    def test_evaluate_predictions(self) -> None:
        preds = [{"answer": "Paris"}, {"answer": "London"}]
        refs = ["Paris", "Berlin"]
        metrics = evaluate_predictions(preds, refs)
        assert metrics["exact_match"] == 0.5


class TestClaimDecomposition:
    def test_simple_split(self) -> None:
        claims = decompose_claims("Paris is the capital. It has 2M people.")
        assert len(claims) == 2
        assert "Paris" in claims[0]

    def test_empty_string(self) -> None:
        assert decompose_claims("") == []

    def test_single_sentence(self) -> None:
        claims = decompose_claims("DNA is a molecule.")
        assert len(claims) == 1

    def test_question_mark_split(self) -> None:
        claims = decompose_claims("What is DNA? It is a molecule.")
        assert len(claims) == 2


class TestFactScore:
    def test_factscore_overlap_fallback(self) -> None:
        """Without nli_fn, falls back to word overlap."""
        ps = [{"id": "0", "text": "Paris is the capital of France"}]
        result = compute_factscore("Paris is the capital of France.", ps)
        assert result["factscore"] > 0.0
        assert result["n_claims"] >= 1

    def test_factscore_with_mock_nli(self) -> None:
        """With a mock nli_fn, should use it."""
        always_entail = lambda premise, hypothesis: 0.9
        ps = [{"id": "0", "text": "anything"}]
        result = compute_factscore(
            "Claim one. Claim two.", ps, nli_fn=always_entail
        )
        assert result["factscore"] == 1.0
        assert result["n_supported"] == 2

    def test_factscore_unsupported(self) -> None:
        never_entail = lambda premise, hypothesis: 0.1
        ps = [{"id": "0", "text": "anything"}]
        result = compute_factscore(
            "Claim one. Claim two.", ps, nli_fn=never_entail
        )
        assert result["factscore"] == 0.0
        assert result["n_supported"] == 0

    def test_factscore_empty_answer(self) -> None:
        result = compute_factscore("", [{"text": "passage"}])
        assert result["n_claims"] == 0
