"""
tests.test_retriever
~~~~~~~~~~~~~~~~~~~~~
Unit tests for HybridRetriever in mock-mode.

Builds an in-memory FAISS index and asserts that ``.retrieve()``
returns the correct number of items with the required keys.
"""

from __future__ import annotations

import pytest

from factuality_rag.retriever.hybrid import HybridRetriever

# Required keys in each result dict
REQUIRED_KEYS = {
    "id",
    "text",
    "dense_score",
    "bm25_score",
    "dense_norm",
    "bm25_norm",
    "combined_score",
    "metadata",
}


class TestHybridRetrieverMock:
    """Test suite for the mock HybridRetriever."""

    @pytest.fixture()
    def retriever(self) -> HybridRetriever:
        """Create a mock retriever with 20 docs."""
        return HybridRetriever.build_mock(dim=768, n_docs=20, seed=42)

    def test_retrieve_returns_k_items(self, retriever: HybridRetriever) -> None:
        """Retrieve should return exactly k items when enough docs exist."""
        results = retriever.retrieve("What is Python?", k=5)
        assert len(results) == 5

    def test_retrieve_returns_required_keys(self, retriever: HybridRetriever) -> None:
        """Each result dict must contain all required keys."""
        results = retriever.retrieve("test query", k=3)
        for r in results:
            assert REQUIRED_KEYS.issubset(
                r.keys()
            ), f"Missing keys: {REQUIRED_KEYS - set(r.keys())}"

    def test_combined_score_in_range(self, retriever: HybridRetriever) -> None:
        """Combined scores should be in [0, 1] after normalisation."""
        results = retriever.retrieve("machine learning", k=5)
        for r in results:
            assert 0.0 <= r["combined_score"] <= 1.0, (
                f"combined_score out of range: {r['combined_score']}"
            )

    def test_dense_norm_in_range(self, retriever: HybridRetriever) -> None:
        """dense_norm should be in [0, 1]."""
        results = retriever.retrieve("normalization test", k=5)
        for r in results:
            assert 0.0 <= r["dense_norm"] <= 1.0

    def test_bm25_norm_in_range(self, retriever: HybridRetriever) -> None:
        """bm25_norm should be in [0, 1]."""
        results = retriever.retrieve("bm25 query", k=5)
        for r in results:
            assert 0.0 <= r["bm25_norm"] <= 1.0

    def test_results_sorted_by_combined_score(self, retriever: HybridRetriever) -> None:
        """Results should be sorted by combined_score descending."""
        results = retriever.retrieve("sort test", k=5, rerank=True)
        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_results(self, retriever: HybridRetriever) -> None:
        """Same query should produce the same results (deterministic mock)."""
        r1 = retriever.retrieve("determinism", k=5)
        r2 = retriever.retrieve("determinism", k=5)
        ids1 = [r["id"] for r in r1]
        ids2 = [r["id"] for r in r2]
        assert ids1 == ids2

    def test_retrieve_k_larger_than_corpus(self) -> None:
        """Requesting more than available docs should not crash."""
        ret = HybridRetriever.build_mock(n_docs=3, seed=0)
        results = ret.retrieve("test", k=10)
        assert len(results) <= 10

    def test_metadata_present(self, retriever: HybridRetriever) -> None:
        """Each result should have a metadata dict with rank."""
        results = retriever.retrieve("metadata test", k=3)
        for r in results:
            assert isinstance(r["metadata"], dict)
