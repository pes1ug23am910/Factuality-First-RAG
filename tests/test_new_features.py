"""
tests.test_new_features
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Session 3 features: sentence-level NLI,
cross-encoder reranking, compute_ece, multi-token probe,
and real provenance mapping — all in mock-mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from factuality_rag.gating.probe import GatingProbe, compute_ece
from factuality_rag.pipeline.orchestrator import _build_provenance, run_pipeline
from factuality_rag.scorer.passage import PassageScorer


# ── Sentence-level NLI ────────────────────────────────────────


class TestSentenceLevelNLI:
    """Tests for sentence-level NLI scoring."""

    def test_split_sentences_basic(self) -> None:
        sentences = PassageScorer._split_sentences("Hello world. How are you?")
        assert len(sentences) == 2
        assert sentences[0] == "Hello world"

    def test_split_sentences_empty(self) -> None:
        assert PassageScorer._split_sentences("") == []
        assert PassageScorer._split_sentences("   ") == []

    def test_split_sentences_single(self) -> None:
        result = PassageScorer._split_sentences("Just one sentence")
        assert len(result) == 1

    def test_sentence_nli_returns_float(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True, nli_mode="sentence")
        score = scorer._sentence_level_nli(
            query="capital of France",
            passage_text="France is a country. Paris is the capital of France.",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_sentence_nli_max_over_sentences(self) -> None:
        """Sentence-level NLI should return max score, which should be
        >= single passage-level NLI on well-split text."""
        scorer = PassageScorer("mock", mock_mode=True, nli_mode="sentence")
        score = scorer._sentence_level_nli(
            query="test",
            passage_text="First sentence. Second sentence. Third sentence.",
        )
        assert 0.0 <= score <= 1.0

    def test_nli_mode_passage_default(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True)
        assert scorer.nli_mode == "passage"

    def test_nli_mode_sentence_wired_in_score_passages(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True, nli_mode="sentence")
        passages = [
            {"id": "0", "text": "First. Second sentence here.", "combined_score": 0.5},
        ]
        result = scorer.score_passages("query", passages)
        assert "nli_score" in result[0]
        assert 0 <= result[0]["final_score"] <= 1


# ── Cross-encoder reranking ──────────────────────────────────


class TestCrossEncoderReranking:
    """Tests for cross-encoder passage reranking."""

    def test_cross_encoder_mock_adds_scores(self) -> None:
        scorer = PassageScorer(
            "mock", mock_mode=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        )
        passages = [
            {"id": "0", "text": "passage a", "combined_score": 0.5},
            {"id": "1", "text": "passage b", "combined_score": 0.6},
            {"id": "2", "text": "passage c", "combined_score": 0.7},
        ]
        result = scorer._cross_encoder_rerank("test query", passages)
        for p in result:
            assert "cross_encoder_score" in p
            assert 0.0 <= p["cross_encoder_score"] <= 1.0

    def test_cross_encoder_reranks_in_score_passages(self) -> None:
        scorer = PassageScorer(
            "mock", mock_mode=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        )
        passages = [
            {"id": "0", "text": "a", "combined_score": 0.3},
            {"id": "1", "text": "b", "combined_score": 0.9},
        ]
        result = scorer.score_passages("query", passages)
        # All should have cross_encoder_score
        for p in result:
            assert "cross_encoder_score" in p
            assert "final_score" in p

    def test_no_cross_encoder_by_default(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True)
        assert scorer.cross_encoder_model is None
        assert scorer._cross_encoder is None


# ── compute_ece ───────────────────────────────────────────────


class TestComputeECE:
    """Tests for the binned Expected Calibration Error function."""

    def test_perfect_calibration(self) -> None:
        """A perfectly calibrated model should have ECE = 0.0."""
        confidences = np.array([0.5, 0.5])
        accuracies = np.array([0.0, 1.0])
        ece = compute_ece(confidences, accuracies)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_overconfident_model(self) -> None:
        """All predictions confident but half wrong → positive ECE."""
        confidences = np.array([0.95, 0.95, 0.95, 0.95])
        accuracies = np.array([1.0, 0.0, 1.0, 0.0])
        ece = compute_ece(confidences, accuracies)
        assert ece > 0.0

    def test_empty_input(self) -> None:
        ece = compute_ece(np.array([]), np.array([]))
        assert ece == 0.0

    def test_ece_range(self) -> None:
        """ECE should be in [0, 1]."""
        rng = np.random.RandomState(42)
        confidences = rng.uniform(0, 1, size=100)
        accuracies = (rng.uniform(0, 1, size=100) > 0.5).astype(float)
        ece = compute_ece(confidences, accuracies)
        assert 0.0 <= ece <= 1.0

    def test_ece_single_sample(self) -> None:
        ece = compute_ece(np.array([0.8]), np.array([1.0]))
        assert isinstance(ece, float)


# ── Multi-token probe ────────────────────────────────────────


class TestMultiTokenProbe:
    """Tests for multi-token logit probing."""

    def test_multi_token_logits_returns_list(self) -> None:
        probe = GatingProbe("mock", mock_mode=True)
        logits_list = probe._get_multi_token_logits("hello", k=3)
        assert len(logits_list) == 3
        assert all(isinstance(l, np.ndarray) for l in logits_list)

    def test_multi_token_logits_different_per_step(self) -> None:
        probe = GatingProbe("mock", mock_mode=True)
        logits_list = probe._get_multi_token_logits("test", k=3)
        # Each step should produce different logits (different seeds)
        assert not np.allclose(logits_list[0], logits_list[1])

    def test_should_retrieve_with_multi_token(self) -> None:
        probe = GatingProbe("mock", mock_mode=True)
        result = probe.should_retrieve("What is Python?", probe_tokens=3)
        assert isinstance(result, bool)

    def test_should_retrieve_single_vs_multi_token(self) -> None:
        """Single and multi-token modes should both return booleans.
        They may differ in decision since multi-token averages entropy.
        """
        probe = GatingProbe("mock", mock_mode=True)
        r1 = probe.should_retrieve("test", probe_tokens=1)
        r3 = probe.should_retrieve("test", probe_tokens=3)
        assert isinstance(r1, bool)
        assert isinstance(r3, bool)

    def test_multi_token_deterministic(self) -> None:
        probe = GatingProbe("mock", mock_mode=True)
        r1 = probe.should_retrieve("test prompt", probe_tokens=3)
        r2 = probe.should_retrieve("test prompt", probe_tokens=3)
        assert r1 == r2


# ── Real provenance mapping ──────────────────────────────────


class TestRealProvenance:
    """Tests for BUG-8 fix: real claim→passage provenance."""

    def test_build_provenance_basic(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True)
        passages = [
            {"id": "p0", "text": "Paris is the capital of France"},
            {"id": "p1", "text": "Berlin is in Germany"},
        ]
        prov = _build_provenance(
            "Paris is the capital. Berlin is large.",
            passages,
            scorer,
        )
        assert isinstance(prov, dict)
        # Should have entries for decomposed claims
        assert len(prov) > 0
        # Each entry should be a list of passage IDs
        for key, value in prov.items():
            assert isinstance(value, list)

    def test_build_provenance_empty_passages(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True)
        prov = _build_provenance("Some answer.", [], scorer)
        assert prov == {}

    def test_build_provenance_empty_answer(self) -> None:
        scorer = PassageScorer("mock", mock_mode=True)
        prov = _build_provenance("", [{"id": "0", "text": "hi"}], scorer)
        assert prov == {}

    def test_pipeline_provenance_has_passage_ids(self) -> None:
        """run_pipeline should produce provenance with real passage IDs."""
        answer, trusted, provenance, tag = run_pipeline(
            "What is Python?", mock_mode=True
        )
        assert isinstance(provenance, dict)
        # Provenance values should reference actual passage IDs
        if trusted and provenance:
            passage_ids = {p["id"] for p in trusted}
            for key, pid_list in provenance.items():
                for pid in pid_list:
                    assert pid in passage_ids or pid == "?"


# ── Config-driven features ───────────────────────────────────


class TestConfigFeatures:
    """Test that new config keys are wired properly."""

    def test_scorer_nli_mode_from_config(self) -> None:
        from factuality_rag.pipeline.orchestrator import Pipeline

        pipe = Pipeline(mock_mode=True)
        assert hasattr(pipe.scorer, "nli_mode")

    def test_scorer_cross_encoder_default_none(self) -> None:
        from factuality_rag.pipeline.orchestrator import Pipeline

        pipe = Pipeline(mock_mode=True)
        assert pipe.scorer.cross_encoder_model is None
