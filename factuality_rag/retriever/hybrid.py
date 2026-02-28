"""
factuality_rag.retriever.hybrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hybrid dense (FAISS) + sparse (BM25 / Pyserini) retriever with
per-query score normalisation and optional re-ranking.

Example (mock-mode)::

    >>> ret = HybridRetriever.build_mock(dim=768, n_docs=20, seed=42)
    >>> results = ret.retrieve("What is Python?", k=5)
    >>> len(results) == 5
    True
    >>> set(results[0].keys()) >= {"id","text","combined_score"}
    True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────
_faiss = None
_SentenceTransformer = None


def _get_faiss():  # type: ignore[no-untyped-def]
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore[import-untyped]
        _faiss = faiss
    return _faiss


def _get_st():  # type: ignore[no-untyped-def]
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class HybridRetriever:
    """Hybrid dense + sparse retriever.

    Args:
        faiss_index_path: Path to the saved FAISS index.
        pyserini_index_path: Path to the Pyserini index directory.
        embed_model: HuggingFace sentence-transformer model name.
        alpha: Weight for dense scores in the combined score
               (``combined = alpha * dense_norm + (1-alpha) * bm25_norm``).
        normalize: Whether to min-max normalise scores to [0, 1].

    Example::

        >>> ret = HybridRetriever.build_mock(dim=768, n_docs=10, seed=42)
        >>> ret.retrieve("hello", k=3)[0]["combined_score"] >= 0.0
        True
    """

    def __init__(
        self,
        faiss_index_path: str,
        pyserini_index_path: str,
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        alpha: float = 0.6,
        normalize: bool = True,
    ) -> None:
        self.faiss_index_path = faiss_index_path
        self.pyserini_index_path = pyserini_index_path
        self.embed_model_name = embed_model
        self.alpha = alpha
        self.normalize = normalize

        # Lazy-loaded state
        self._faiss_index: Any = None
        self._id_map: List[str] = []
        self._texts: List[str] = []
        self._embed_model: Any = None
        self._mock_mode: bool = False
        self._mock_embeddings: Optional[np.ndarray] = None
        self._dim: int = 768

    # ── Factory helpers ───────────────────────────────────────

    @classmethod
    def build_mock(
        cls,
        dim: int = 768,
        n_docs: int = 20,
        seed: int = 42,
        alpha: float = 0.6,
    ) -> "HybridRetriever":
        """Create an **in-memory** mock retriever for testing.

        Args:
            dim: Embedding dimensionality.
            n_docs: Number of synthetic documents.
            seed: Random seed for reproducibility.
            alpha: Dense weight.

        Returns:
            A ready-to-query ``HybridRetriever`` in mock-mode.

        Example::

            >>> ret = HybridRetriever.build_mock(n_docs=5)
            >>> len(ret.retrieve("query", k=3))
            3
        """
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(n_docs, dim).astype(np.float32)

        faiss = _get_faiss()
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        instance = cls.__new__(cls)
        instance.faiss_index_path = ":memory:"
        instance.pyserini_index_path = ":memory:"
        instance.embed_model_name = "mock"
        instance.alpha = alpha
        instance.normalize = True
        instance._faiss_index = index
        instance._id_map = [f"doc_{i}" for i in range(n_docs)]
        instance._texts = [
            f"Mock passage {i}. This is a test document about topic {i}."
            for i in range(n_docs)
        ]
        instance._embed_model = None
        instance._mock_mode = True
        instance._mock_embeddings = embeddings
        instance._dim = dim
        return instance

    # ── Loading helpers ───────────────────────────────────────

    def _load_index(self) -> None:
        """Load FAISS index and id mapping from disk.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        if self._faiss_index is not None:
            return
        faiss = _get_faiss()
        p = Path(self.faiss_index_path)
        if not p.exists():
            raise FileNotFoundError(f"FAISS index not found: {p}")
        self._faiss_index = faiss.read_index(str(p))

        id_map_path = p.with_suffix(".ids.json")
        if id_map_path.exists():
            with open(id_map_path, encoding="utf-8") as f:
                self._id_map = json.load(f)

        # Try to load texts from the corpus JSONL next to the index
        corpus_path = p.with_suffix(".jsonl")
        if corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self._texts.append(obj.get("text", ""))
        logger.info("Loaded FAISS index: %d vectors", self._faiss_index.ntotal)

    def _get_embed_model(self) -> Any:
        """Lazy-load the sentence-transformer model.

        Returns:
            A ``SentenceTransformer`` instance.
        """
        if self._embed_model is None and not self._mock_mode:
            ST = _get_st()
            self._embed_model = ST(self.embed_model_name)
        return self._embed_model

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve(
        self, query: str, k: int = 10, rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve top-*k* passages by hybrid dense+sparse scoring.

        Returns list of dicts with keys:
        ``{id, text, dense_score, bm25_score, dense_norm, bm25_norm,
        combined_score, metadata}``.

        Normalisation: ``dense_norm`` and ``bm25_norm`` are min-max scaled
        to [0, 1] per query. ``combined_score = alpha * dense_norm +
        (1-alpha) * bm25_norm``.

        Args:
            query: Natural language query string.
            k: Number of results.
            rerank: If ``True``, sort by ``combined_score`` desc.

        Returns:
            List of result dicts.

        Example::

            >>> ret = HybridRetriever.build_mock(n_docs=10)
            >>> res = ret.retrieve("test", k=5)
            >>> len(res)
            5
            >>> all(0 <= r["combined_score"] <= 1 for r in res)
            True
        """
        self._load_index()

        # ── Dense retrieval ───────────────────────────────────
        if self._mock_mode:
            rng = np.random.RandomState(abs(hash(query)) % (2**31))
            q_vec = rng.randn(1, self._dim).astype(np.float32)
        else:
            model = self._get_embed_model()
            q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

        distances, indices = self._faiss_index.search(q_vec, k)
        distances = distances[0]
        indices = indices[0]

        # ── BM25 retrieval (mock or Pyserini) ─────────────────
        bm25_scores = self._bm25_search(query, k)

        # ── Merge ─────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            if idx < 0:
                continue
            doc_id = self._id_map[idx] if idx < len(self._id_map) else str(idx)
            text = self._texts[idx] if idx < len(self._texts) else ""
            dense_score = float(-dist)  # FAISS L2 → negate for similarity
            bm25_score = bm25_scores.get(doc_id, 0.0)
            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "dense_score": dense_score,
                    "bm25_score": bm25_score,
                    "dense_norm": 0.0,  # filled below
                    "bm25_norm": 0.0,
                    "combined_score": 0.0,
                    "metadata": {"rank": rank},
                }
            )

        # ── Normalise & combine ───────────────────────────────
        if results and self.normalize:
            d_scores = [r["dense_score"] for r in results]
            b_scores = [r["bm25_score"] for r in results]
            d_min, d_max = min(d_scores), max(d_scores)
            b_min, b_max = min(b_scores), max(b_scores)

            for r in results:
                r["dense_norm"] = (
                    (r["dense_score"] - d_min) / (d_max - d_min)
                    if d_max > d_min
                    else 0.5
                )
                r["bm25_norm"] = (
                    (r["bm25_score"] - b_min) / (b_max - b_min)
                    if b_max > b_min
                    else 0.5
                )
                r["combined_score"] = (
                    self.alpha * r["dense_norm"]
                    + (1 - self.alpha) * r["bm25_norm"]
                )

        if rerank:
            results.sort(key=lambda r: r["combined_score"], reverse=True)

        return results[:k]

    # ── BM25 backend ──────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> Dict[str, float]:
        """Run BM25 search via Pyserini or return mock scores.

        Uses a real ``LuceneSearcher`` when *pyserini_index_path*
        points to a valid Lucene index.  Falls back to mock scores
        with a warning otherwise.

        Args:
            query: Query string.
            k: Number of results.

        Returns:
            Mapping from doc id to BM25 score.
        """
        if self._mock_mode:
            rng = np.random.RandomState(abs(hash(query)) % (2**31) + 1)
            return {
                doc_id: float(rng.uniform(0, 10))
                for doc_id in self._id_map[:k]
            }

        # Real Pyserini BM25 search
        try:
            from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]

            idx_path = Path(self.pyserini_index_path)
            if not idx_path.exists():
                logger.warning(
                    "Pyserini index not found at '%s'; falling back to empty BM25.",
                    idx_path,
                )
                return {}

            searcher = LuceneSearcher(str(idx_path))
            hits = searcher.search(query, k)
            return {hit.docid: float(hit.score) for hit in hits}
        except ImportError:
            logger.warning(
                "Pyserini not installed; BM25 search disabled. "
                "Install with: pip install pyserini"
            )
            return {}
        except Exception as exc:
            logger.warning("Pyserini BM25 search failed: %s", exc)
            return {}
