"""
tests.test_learned_scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the LearnedScorer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from factuality_rag.scorer.learned_scorer import LearnedScorer


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def sample_data():
    """Simple linearly-separable training set."""
    X = [
        [0.9, 0.6, 0.8],  # positive
        [0.85, 0.5, 0.7],  # positive
        [0.8, 0.4, 0.9],  # positive
        [0.1, 0.1, 0.2],  # negative
        [0.15, 0.05, 0.1],  # negative
        [0.2, 0.15, 0.3],  # negative
    ]
    y = [1, 1, 1, 0, 0, 0]
    return np.array(X), np.array(y)


# ── Logistic regression tests ────────────────────────────────


class TestLearnedScorerLogreg:
    def test_fit_and_predict(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        probs = ls.predict_proba(X)
        assert probs.shape == (6,)
        assert all(0 <= p <= 1 for p in probs)

    def test_separates_classes(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        probs = ls.predict_proba(X)
        # Positive examples should have higher proba than negatives
        assert probs[:3].mean() > probs[3:].mean()

    def test_predict_before_fit_raises(self):
        ls = LearnedScorer("logreg")
        with pytest.raises(RuntimeError, match="not fitted"):
            ls.predict_proba([[0.5, 0.5, 0.5]])

    def test_invalid_shape_raises(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        with pytest.raises(ValueError, match="Expected X shape"):
            ls.fit([[0.5, 0.5]], [1])  # 2 features instead of 3

    def test_repr(self):
        ls = LearnedScorer("logreg")
        assert "logreg" in repr(ls)


# ── MLP tests ────────────────────────────────────────────────


class TestLearnedScorerMLP:
    def test_fit_and_predict(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("mlp")
        ls.fit(X, y)
        probs = ls.predict_proba(X)
        assert probs.shape == (6,)
        assert all(0 <= p <= 1 for p in probs)

    def test_repr(self):
        ls = LearnedScorer("mlp")
        assert "mlp" in repr(ls)


# ── Persistence tests ────────────────────────────────────────


class TestLearnedScorerPersistence:
    def test_save_load_logreg(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        probs_before = ls.predict_proba(X)

        with tempfile.TemporaryDirectory() as d:
            ls.save(d)
            assert Path(d, "model.pkl").exists()
            assert Path(d, "metadata.json").exists()

            ls2 = LearnedScorer.load(d)
            probs_after = ls2.predict_proba(X)
            np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)

    def test_save_load_mlp(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("mlp")
        ls.fit(X, y)

        with tempfile.TemporaryDirectory() as d:
            ls.save(d)
            ls2 = LearnedScorer.load(d)
            assert ls2._fitted
            assert ls2.classifier_type == "mlp"

    def test_metadata_content(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)

        with tempfile.TemporaryDirectory() as d:
            ls.save(d)
            with open(Path(d, "metadata.json")) as f:
                meta = json.load(f)
            assert meta["classifier_type"] == "logreg"
            assert "learned_weights" in meta
            assert set(meta["learned_weights"].keys()) == {
                "nli_score", "overlap_score", "retriever_score_norm"
            }


# ── Evaluation tests ─────────────────────────────────────────


class TestLearnedScorerEvaluation:
    def test_evaluate_returns_metrics(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        metrics = ls.evaluate(X, y)
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_perfect_classification(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        metrics = ls.evaluate(X, y)
        # Linearly separable data → should get near-perfect accuracy
        assert metrics["accuracy"] >= 0.8


# ── score_passages tests ─────────────────────────────────────


class TestLearnedScorerScorePassages:
    def test_adds_learned_score(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)

        passages = [
            {"nli_score": 0.9, "overlap_score": 0.5, "combined_score": 0.8},
            {"nli_score": 0.1, "overlap_score": 0.1, "combined_score": 0.2},
        ]
        result = ls.score_passages(passages)
        for p in result:
            assert "learned_score" in p
            assert 0 <= p["learned_score"] <= 1

    def test_empty_passages(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        assert ls.score_passages([]) == []

    def test_high_vs_low(self, sample_data):
        X, y = sample_data
        ls = LearnedScorer("logreg")
        ls.fit(X, y)
        passages = [
            {"nli_score": 0.9, "overlap_score": 0.6, "combined_score": 0.85},
            {"nli_score": 0.1, "overlap_score": 0.05, "combined_score": 0.15},
        ]
        result = ls.score_passages(passages)
        assert result[0]["learned_score"] > result[1]["learned_score"]
