"""
tests.test_scorer
~~~~~~~~~~~~~~~~~~
Unit tests for PassageScorer in mock-mode.
"""

from __future__ import annotations

import pytest

from factuality_rag.scorer.passage import PassageScorer


class TestPassageScorerMock:
    """Test suite for mock-mode PassageScorer."""

    @pytest.fixture()
    def scorer(self) -> PassageScorer:
        return PassageScorer(
            nli_model_hf="mock-nli",
            mock_mode=True,
        )

    def test_score_passages_adds_keys(self, scorer: PassageScorer) -> None:
        passages = [
            {"id": "0", "text": "Paris is the capital of France", "combined_score": 0.8},
            {"id": "1", "text": "Berlin is the capital of Germany", "combined_score": 0.5},
        ]
        result = scorer.score_passages("capital of France", passages)
        for p in result:
            assert "nli_score" in p
            assert "overlap_score" in p
            assert "final_score" in p

    def test_final_score_in_range(self, scorer: PassageScorer) -> None:
        passages = [
            {"id": "0", "text": "hello world", "combined_score": 0.7},
        ]
        result = scorer.score_passages("hello", passages)
        for p in result:
            assert 0.0 <= p["final_score"] <= 1.0

    def test_token_overlap(self) -> None:
        score = PassageScorer._token_overlap("hello world", "hello there world")
        assert 0.0 < score <= 1.0

    def test_empty_passages(self, scorer: PassageScorer) -> None:
        result = scorer.score_passages("query", [])
        assert result == []

    def test_nli_argument_order_docstring(self) -> None:
        """BUG-2 regression: _nli_entailment docstring must say
        premise=passage, hypothesis=query."""
        import inspect

        src = inspect.getsource(PassageScorer._nli_entailment)
        assert "passage" in src.lower() and "premise" in src.lower()

    def test_score_passages_calls_nli_with_correct_order(self) -> None:
        """BUG-2 regression: verify score_passages passes
        passage_text as premise and query as hypothesis to NLI."""
        calls = []

        class SpyScorer(PassageScorer):
            def _nli_entailment(self, premise: str, hypothesis: str) -> float:
                calls.append((premise, hypothesis))
                return 0.5

        scorer = SpyScorer(mock_mode=True)
        passages = [{"id": "0", "text": "The passage text", "combined_score": 0.5}]
        scorer.score_passages("The query", passages)

        assert len(calls) == 1
        # premise should be the passage text, hypothesis should be the query
        assert calls[0][0] == "The passage text"
        assert calls[0][1] == "The query"
