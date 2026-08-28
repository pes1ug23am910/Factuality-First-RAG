"""
tests.test_retriever
~~~~~~~~~~~~~~~~~~~~~
Unit tests for HybridRetriever in mock-mode.

Builds an in-memory FAISS index and asserts that ``.retrieve()``
returns the correct number of items with the required keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import factuality_rag.retriever.hybrid as hybrid
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
            assert REQUIRED_KEYS.issubset(r.keys()), (
                f"Missing keys: {REQUIRED_KEYS - set(r.keys())}"
            )

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

    def test_reuses_corpus_text_lookup_across_queries(self, retriever: HybridRetriever) -> None:
        """The corpus-sized ID lookup should be allocated only once."""
        lookup = retriever._text_by_id

        retriever.retrieve("first query", k=3)
        retriever.retrieve("second query", k=3)

        assert retriever._text_by_id is lookup

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


def _fixed_retriever(
    metric_type: int,
    distances: list[float],
    indices: list[int],
    *,
    alpha: float = 0.4,
    normalize: bool = True,
) -> HybridRetriever:
    class FixedIndex:
        ntotal = 3

        def __init__(self) -> None:
            self.metric_type = metric_type

        def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            assert query.shape == (1, 2)
            return (
                np.asarray([distances], dtype=np.float32),
                np.asarray([indices], dtype=np.int64),
            )

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.faiss_index_path = ":memory:"
    retriever.pyserini_index_path = ":memory:"
    retriever.embed_model_name = "mock"
    retriever.alpha = alpha
    retriever.normalize = normalize
    retriever._faiss_index = FixedIndex()
    retriever._id_map = ["dense-a", "dense-b", "sparse-only"]
    retriever._texts = ["dense a", "dense b", "sparse only"]
    retriever._embed_model = None
    retriever._mock_mode = True
    retriever._mock_embeddings = np.zeros((3, 2), dtype=np.float32)
    retriever._dim = 2
    retriever._lucene_searcher = None
    return retriever


def test_hybrid_fusion_uses_union_including_sparse_only_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faiss = hybrid._get_faiss()
    retriever = _fixed_retriever(faiss.METRIC_L2, [0.1, 0.2], [0, 1])
    monkeypatch.setattr(
        retriever,
        "_bm25_search",
        lambda query, k: {"sparse-only": 10.0, "dense-b": 1.0},
    )

    results = retriever.retrieve("query", k=3)

    assert {result["id"] for result in results} == {"dense-a", "dense-b", "sparse-only"}
    assert results[0]["id"] == "sparse-only"
    assert results[0]["metadata"]["dense_rank"] is None


@pytest.mark.parametrize(
    "metric_name,distances,expected",
    [
        ("l2", [0.1, 0.4], [-0.1, -0.4]),
        ("ip", [0.9, 0.4], [0.9, 0.4]),
    ],
)
def test_dense_score_direction_matches_faiss_metric(
    monkeypatch: pytest.MonkeyPatch,
    metric_name: str,
    distances: list[float],
    expected: list[float],
) -> None:
    faiss = hybrid._get_faiss()
    metric = faiss.METRIC_L2 if metric_name == "l2" else faiss.METRIC_INNER_PRODUCT
    retriever = _fixed_retriever(metric, distances, [0, 1])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: {"dense-a": 1.0})

    by_id = {item["id"]: item for item in retriever.retrieve("query", k=3)}

    assert by_id["dense-a"]["dense_score"] == pytest.approx(expected[0])
    assert by_id["dense-b"]["dense_score"] == pytest.approx(expected[1])


def test_non_normalized_fusion_still_computes_combined_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faiss = hybrid._get_faiss()
    retriever = _fixed_retriever(
        faiss.METRIC_INNER_PRODUCT,
        [0.8, 0.2],
        [0, 1],
        alpha=0.25,
        normalize=False,
    )
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: {"dense-a": 2.0})

    result = {item["id"]: item for item in retriever.retrieve("query", k=3)}["dense-a"]

    assert result["combined_score"] == pytest.approx(0.25 * 0.8 + 0.75 * 2.0)


def test_non_normalized_l2_missing_dense_component_cannot_beat_dense_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faiss = hybrid._get_faiss()
    retriever = _fixed_retriever(
        faiss.METRIC_L2,
        [0.1, 0.2],
        [0, 1],
        alpha=1.0,
        normalize=False,
    )
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: {"sparse-only": 10.0})

    results = retriever.retrieve("query", k=3)

    assert results[0]["id"] == "dense-a"
    assert {item["id"]: item["combined_score"] for item in results}["sparse-only"] <= -0.2


def test_unknown_faiss_metric_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _fixed_retriever(999999, [0.8], [0])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: {"dense-a": 1.0})

    with pytest.raises(ValueError, match="unsupported or unknown FAISS metric_type"):
        retriever.retrieve("query", k=1)


class _LoadOnlyFaiss:
    @staticmethod
    def read_index(path: str) -> Any:
        assert Path(path).is_file()

        class Index:
            ntotal = 2

        return Index()


def _write_index_triplet(tmp_path: Path, corpus_records: list[dict[str, str]]) -> Path:
    index_path = tmp_path / "bound.faiss"
    index_path.write_bytes(b"fake-index")
    index_path.with_suffix(".ids.json").write_text(json.dumps(["doc-0", "doc-1"]), encoding="utf-8")
    index_path.with_suffix(".jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in corpus_records),
        encoding="utf-8",
    )
    return index_path


def test_real_loader_rejects_corpus_id_order_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _write_index_triplet(
        tmp_path,
        [
            {"id": "doc-1", "text": "wrong order"},
            {"id": "doc-0", "text": "wrong order"},
        ],
    )
    monkeypatch.setattr(hybrid, "_get_faiss", lambda: _LoadOnlyFaiss())
    retriever = HybridRetriever(str(index_path), str(tmp_path / "lucene"))

    with pytest.raises(ValueError, match="ID/order mismatch"):
        retriever._load_index()


def test_real_loader_never_uses_hostile_cwd_corpus_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_path = index_dir / "bound.faiss"
    index_path.write_bytes(b"fake-index")
    index_path.with_suffix(".ids.json").write_text(json.dumps(["doc-0", "doc-1"]), encoding="utf-8")
    hostile_data = tmp_path / "hostile" / "data"
    hostile_data.mkdir(parents=True)
    (hostile_data / "wiki_100000_chunks.jsonl").write_text(
        '{"id":"doc-0","text":"malicious"}\n{"id":"doc-1","text":"malicious"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(hostile_data.parent)
    monkeypatch.setattr(hybrid, "_get_faiss", lambda: _LoadOnlyFaiss())
    retriever = HybridRetriever(str(index_path), str(tmp_path / "lucene"))

    with pytest.raises(FileNotFoundError, match="bound FAISS corpus sidecar"):
        retriever._load_index()


def test_explicit_bound_corpus_loads_only_when_ids_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _write_index_triplet(
        tmp_path,
        [
            {"id": "doc-0", "text": "first"},
            {"id": "doc-1", "text": "second"},
        ],
    )
    corpus_path = index_path.with_suffix(".jsonl")
    monkeypatch.setattr(hybrid, "_get_faiss", lambda: _LoadOnlyFaiss())
    retriever = HybridRetriever(
        str(index_path), str(tmp_path / "lucene"), corpus_path=str(corpus_path)
    )

    retriever._load_index()

    assert retriever._id_map == ["doc-0", "doc-1"]
    assert retriever._texts == ["first", "second"]
