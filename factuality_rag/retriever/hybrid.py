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

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional
import unicodedata

import numpy as np

from factuality_rag.determinism import stable_seed

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────
_faiss = None
_SentenceTransformer = None

_PYSERINI_PROTOCOL_VERSION = 1
_PYSERINI_WORKER_TIMEOUT_SECONDS = 60.0
_PYSERINI_WORKER_MODULE = "factuality_rag.retriever.pyserini_worker"
_PYSERINI_MAX_K = 1_000
_PYSERINI_MAX_QUERY_BYTES = 100_000
_PYSERINI_MAX_REQUEST_BYTES = 200_000
_PYSERINI_MAX_STDOUT_BYTES = 10_000_000
_PYSERINI_MAX_STDERR_BYTES = 1_000_000
_JAVA_HOME_OVERRIDE = "FACTUALITY_RAG_JAVA_HOME"
_NATIVE_ERROR_MARKERS = (
    "windows fatal exception",
    "access violation",
    "exception_access_violation",
    "fatal python error",
    "a fatal error has been detected by the java runtime environment",
    "jre fatal",
    "jni error",
    "problematic frame",
    "segmentation fault",
    "sigsegv",
    "hs_err_pid",
)
_ALLOWED_WORKER_STDERR_LINES = {
    "WARNING: Using incubator modules: jdk.incubator.vector",
    (
        "INFO: Using MemorySegmentIndexInput with Java 21; to disable start with "
        "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
    ),
}
_LUCENE_STARTUP_LOG_PATTERN = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"[ 0-9][0-9], [0-9]{4} [0-9]{1,2}:[0-9]{2}:[0-9]{2} (?:AM|PM) "
    r"org\.apache\.lucene\.store\.MemorySegmentIndexInputProvider <init>$"
)


class PyseriniWorkerError(RuntimeError):
    """Raised when the isolated Windows Pyserini request cannot be trusted."""


class BM25BackendError(RuntimeError):
    """Raised when a configured real BM25 backend cannot produce trusted results."""


def _java_major(java_home: Path) -> int:
    try:
        release = (java_home / "release").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PyseriniWorkerError("Selected JAVA_HOME has no readable release metadata") from exc
    match = re.search(r'^JAVA_VERSION="([0-9]+)(?:[._][^"]*)?"$', release, flags=re.MULTILINE)
    if match is None:
        raise PyseriniWorkerError("Selected JAVA_HOME has invalid release metadata")
    return int(match.group(1))


def _worker_environment() -> Dict[str, str]:
    """Build an isolated Java 21 environment without mutating ``os.environ``."""
    child_environment = dict(os.environ)
    selected_home = child_environment.get(_JAVA_HOME_OVERRIDE) or child_environment.get("JAVA_HOME")
    if not selected_home:
        raise PyseriniWorkerError(
            "Java 21 is required; set FACTUALITY_RAG_JAVA_HOME to a Java 21 installation"
        )
    try:
        java_home = Path(selected_home).resolve(strict=True)
    except OSError as exc:
        raise PyseriniWorkerError("Selected JAVA_HOME does not exist") from exc
    java_name = "java.exe" if sys.platform == "win32" else "java"
    java_executable = java_home / "bin" / java_name
    if not java_home.is_dir() or not java_executable.is_file() or _java_major(java_home) != 21:
        raise PyseriniWorkerError("The isolated Pyserini worker requires a Java 21 JAVA_HOME")

    child_environment["JAVA_HOME"] = str(java_home)
    child_environment["PATH"] = (
        str(java_home / "bin") + os.pathsep + child_environment.get("PATH", "")
    )
    child_environment["PYTHONUTF8"] = "1"
    for variable in ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS", "CLASSPATH"):
        child_environment.pop(variable, None)
    return child_environment


def _reject_duplicate_json_keys(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_worker_response(stdout: str, k: int, request_sha256: str) -> Dict[str, float]:
    """Validate the worker's single-line JSON response."""
    if not isinstance(stdout, str) or not stdout:
        raise ValueError("Pyserini worker produced no JSON response")

    payload = stdout[:-1] if stdout.endswith("\n") else stdout
    if not payload or "\n" in payload or "\r" in payload or payload != payload.strip():
        raise ValueError("Pyserini worker produced extra stdout")

    value = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_float=_parse_finite_json_float,
        parse_constant=_reject_non_finite_json,
    )
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "request_sha256",
        "hits",
    }:
        raise ValueError("Pyserini worker response has an invalid top-level schema")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != _PYSERINI_PROTOCOL_VERSION
    ):
        raise ValueError("Pyserini worker response uses an unsupported schema version")
    if value["request_sha256"] != request_sha256:
        raise ValueError("Pyserini worker response does not match its request")

    hits = value["hits"]
    if not isinstance(hits, list) or len(hits) > k:
        raise ValueError("Pyserini worker response has an invalid hit list")

    scores: Dict[str, float] = {}
    for hit in hits:
        if not isinstance(hit, dict) or set(hit) != {"docid", "score"}:
            raise ValueError("Pyserini worker response contains an invalid hit")
        doc_id = hit["docid"]
        raw_score = hit["score"]
        if (
            not isinstance(doc_id, str)
            or not doc_id
            or doc_id != doc_id.strip()
            or any(unicodedata.category(character).startswith("C") for character in doc_id)
        ):
            raise ValueError("Pyserini worker response contains an invalid document id")
        if doc_id in scores:
            raise ValueError(f"Pyserini worker returned duplicate document id: {doc_id!r}")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("Pyserini worker response contains a non-numeric score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Pyserini worker response contains a non-finite score")
        scores[doc_id] = score
    return scores


def _search_pyserini_in_worker(index_path: Path, query: str, k: int) -> Dict[str, float]:
    """Run Pyserini behind a process boundary on Windows."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= _PYSERINI_MAX_K:
        raise ValueError(f"k must be an integer in [1, {_PYSERINI_MAX_K}]")
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if len(query.encode("utf-8")) > _PYSERINI_MAX_QUERY_BYTES:
        raise ValueError("query exceeds the Pyserini worker protocol limit")

    request = {
        "schema_version": _PYSERINI_PROTOCOL_VERSION,
        "index_path": str(index_path.resolve(strict=True)),
        "query": query,
        "k": k,
    }
    request_json = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    request_bytes = request_json.encode("utf-8")
    if len(request_bytes) > _PYSERINI_MAX_REQUEST_BYTES:
        raise ValueError("request exceeds the Pyserini worker protocol limit")
    request_sha256 = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    try:
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_file,
            tempfile.TemporaryFile(mode="w+b") as stderr_file,
        ):
            completed = subprocess.run(
                [sys.executable, "-m", _PYSERINI_WORKER_MODULE],
                input=request_bytes + b"\n",
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=_PYSERINI_WORKER_TIMEOUT_SECONDS,
                check=False,
                shell=False,
                env=_worker_environment(),
            )
            stdout_size = stdout_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.seek(0, os.SEEK_END)
            if stdout_size > _PYSERINI_MAX_STDOUT_BYTES:
                raise PyseriniWorkerError("Pyserini worker stdout exceeded the protocol limit")
            if stderr_size > _PYSERINI_MAX_STDERR_BYTES:
                raise PyseriniWorkerError("Pyserini worker stderr exceeded the protocol limit")
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_bytes = stdout_file.read()
            stderr_bytes = stderr_file.read()
    except subprocess.TimeoutExpired as exc:
        raise PyseriniWorkerError("Pyserini worker timed out") from exc
    except (OSError, UnicodeError) as exc:
        raise PyseriniWorkerError("Pyserini worker could not complete safely") from exc

    if not isinstance(stderr_bytes, bytes) or not isinstance(stdout_bytes, bytes):
        raise PyseriniWorkerError("Pyserini worker returned a non-binary protocol response")
    try:
        stderr = stderr_bytes.decode("utf-8", errors="strict")
        stdout = stdout_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PyseriniWorkerError("Pyserini worker returned invalid UTF-8") from exc
    lowered_stderr = stderr.casefold()
    marker = next((item for item in _NATIVE_ERROR_MARKERS if item in lowered_stderr), None)
    if marker is not None:
        raise PyseriniWorkerError(f"Pyserini worker reported a native failure marker: {marker}")
    if completed.returncode != 0:
        raise PyseriniWorkerError(f"Pyserini worker exited with status {completed.returncode}")
    stderr_lines = [
        line for line in stderr.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line
    ]
    if any(
        line not in _ALLOWED_WORKER_STDERR_LINES
        and _LUCENE_STARTUP_LOG_PATTERN.fullmatch(line) is None
        for line in stderr_lines
    ):
        raise PyseriniWorkerError("Pyserini worker produced unexpected stderr")
    try:
        return _parse_worker_response(stdout, k, request_sha256)
    except (OverflowError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PyseriniWorkerError("Pyserini worker returned an invalid response") from exc


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
        corpus_path: Optional explicit ordered JSONL corpus bound to the FAISS
            ID mapping. When omitted, the index-adjacent ``.jsonl`` is required.

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
        *,
        corpus_path: Optional[str] = None,
    ) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be numeric")
        if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        if type(normalize) is not bool:
            raise TypeError("normalize must be boolean")
        if corpus_path is not None and (
            not isinstance(corpus_path, str)
            or not corpus_path
            or corpus_path != corpus_path.strip()
        ):
            raise ValueError("corpus_path must be None or a non-empty trimmed string")
        self.faiss_index_path = faiss_index_path
        self.pyserini_index_path = pyserini_index_path
        self.corpus_path = corpus_path
        self.embed_model_name = embed_model
        self.alpha = alpha
        self.normalize = normalize

        # Lazy-loaded state
        self._faiss_index: Any = None
        self._id_map: List[str] = []
        self._texts: List[str] = []
        self._text_by_id: Dict[str, str] = {}
        self._embed_model: Any = None
        self._mock_mode: bool = False
        self._mock_embeddings: Optional[np.ndarray] = None
        self._dim: int = 768
        self._lucene_searcher: Any = None

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
        instance.corpus_path = None
        instance.embed_model_name = "mock"
        instance.alpha = alpha
        instance.normalize = True
        instance._faiss_index = index
        instance._id_map = [f"doc_{i}" for i in range(n_docs)]
        instance._texts = [
            f"Mock passage {i}. This is a test document about topic {i}." for i in range(n_docs)
        ]
        instance._text_by_id = dict(zip(instance._id_map, instance._texts))
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

        # A real index is one typed triplet: vectors, ordered IDs, and an
        # ordered corpus.  Never search the caller's working directory for a
        # plausible same-length corpus.
        id_map_path = p.with_suffix(".ids.json")
        if not id_map_path.is_file():
            raise FileNotFoundError(f"FAISS ID mapping not found: {id_map_path}")
        with id_map_path.open(encoding="utf-8") as id_file:
            raw_id_map = json.load(id_file)
        if not isinstance(raw_id_map, list):
            raise ValueError("FAISS ID mapping must be a JSON list")
        self._id_map = []
        seen_ids = set()
        for index, value in enumerate(raw_id_map):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"FAISS ID mapping entry {index} must be a non-empty string")
            if value in seen_ids:
                raise ValueError(f"FAISS ID mapping contains duplicate document id: {value!r}")
            seen_ids.add(value)
            self._id_map.append(value)

        corpus_path = (
            Path(self.corpus_path) if self.corpus_path is not None else p.with_suffix(".jsonl")
        )
        if not corpus_path.is_file():
            raise FileNotFoundError(
                "bound FAISS corpus sidecar not found; provide corpus_path or place the "
                f"ordered JSONL beside the index: {corpus_path}"
            )
        self._texts = []
        with corpus_path.open(encoding="utf-8") as corpus_file:
            for index, line in enumerate(corpus_file):
                try:
                    record = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"FAISS corpus line {index + 1} is not strict JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"FAISS corpus line {index + 1} must contain an object")
                if index >= len(self._id_map):
                    raise ValueError("FAISS corpus has more records than the ID mapping")
                document_id = record.get("id")
                if document_id != self._id_map[index]:
                    raise ValueError(
                        f"FAISS corpus ID/order mismatch at record {index}: "
                        f"{document_id!r} != {self._id_map[index]!r}"
                    )
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"FAISS corpus record {index} has invalid text")
                self._texts.append(text)
        if len(self._texts) != len(self._id_map):
            raise ValueError("FAISS corpus and ID mapping record counts differ")
        self._text_by_id = dict(zip(self._id_map, self._texts))
        if len(self._text_by_id) != len(self._id_map):
            raise ValueError("FAISS id mapping contains duplicate document ids")

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

    def retrieve(self, query: str, k: int = 10, rerank: bool = True) -> List[Dict[str, Any]]:
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
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-blank string")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= _PYSERINI_MAX_K:
            raise ValueError(f"k must be an integer in [1, {_PYSERINI_MAX_K}]")
        if type(rerank) is not bool:
            raise TypeError("rerank must be exactly bool")

        self._load_index()
        index_size = getattr(self._faiss_index, "ntotal", None)
        if (
            isinstance(index_size, bool)
            or not isinstance(index_size, int)
            or index_size < 0
            or len(self._id_map) != index_size
            or len(self._texts) != index_size
        ):
            raise ValueError("FAISS vectors, document IDs, and passage texts must align exactly")

        # ── Dense retrieval ───────────────────────────────────
        if self._mock_mode:
            rng = np.random.RandomState(stable_seed("hybrid_retriever.mock_dense", query))
            q_vec = rng.randn(1, self._dim).astype(np.float32)
        else:
            model = self._get_embed_model()
            q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

        faiss = _get_faiss()
        metric_type = getattr(self._faiss_index, "metric_type", None)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            # The supported IP builder stores L2-normalized document vectors,
            # so normalize the query to preserve cosine-similarity semantics.
            faiss.normalize_L2(q_vec)
            dense_metric = "inner_product"
        elif metric_type == faiss.METRIC_L2:
            dense_metric = "squared_l2"
        else:
            raise ValueError(f"unsupported or unknown FAISS metric_type: {metric_type!r}")

        distances, indices = self._faiss_index.search(q_vec, k)
        distances = distances[0]
        indices = indices[0]

        # ── BM25 retrieval (mock or Pyserini) ─────────────────
        bm25_scores = self._bm25_search(query, k)

        # ── Union and fuse ────────────────────────────────────
        dense_candidates: Dict[str, Dict[str, Any]] = {}
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            if idx < 0:
                continue
            doc_id = self._id_map[idx] if idx < len(self._id_map) else str(idx)
            text = self._texts[idx] if idx < len(self._texts) else ""
            raw_distance = float(dist)
            if not math.isfinite(raw_distance):
                raise ValueError("FAISS returned a non-finite score")
            dense_score = raw_distance if dense_metric == "inner_product" else -raw_distance
            if doc_id in dense_candidates:
                raise ValueError(f"FAISS returned duplicate document id: {doc_id!r}")
            dense_candidates[doc_id] = {
                "text": text,
                "score": dense_score,
                "rank": rank,
            }

        # Cache this corpus-sized lookup at load time. Older in-memory test
        # fixtures may predate the attribute, so initialize it lazily once.
        id_to_text = getattr(self, "_text_by_id", None)
        if not id_to_text:
            id_to_text = dict(zip(self._id_map, self._texts))
            if len(id_to_text) != len(self._id_map):
                raise ValueError("FAISS id mapping contains duplicate document ids")
            self._text_by_id = id_to_text
        for doc_id, score in bm25_scores.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("BM25 returned an invalid document id")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("BM25 returned a non-numeric score")
            if not math.isfinite(float(score)):
                raise ValueError("BM25 returned a non-finite score")
            if doc_id not in id_to_text:
                raise ValueError(
                    f"BM25 document id {doc_id!r} is absent from the bound FAISS corpus mapping"
                )

        dense_order = list(dense_candidates)
        sparse_only_order = [doc_id for doc_id in bm25_scores if doc_id not in dense_candidates]
        candidate_order = dense_order + sparse_only_order

        dense_norms: Dict[str, float] = {}
        bm25_norms: Dict[str, float] = {}
        if self.normalize:
            if dense_candidates:
                dense_values = [float(item["score"]) for item in dense_candidates.values()]
                dense_min, dense_max = min(dense_values), max(dense_values)
                dense_norms = {
                    doc_id: (
                        (float(item["score"]) - dense_min) / (dense_max - dense_min)
                        if dense_max > dense_min
                        else 1.0
                    )
                    for doc_id, item in dense_candidates.items()
                }
            if bm25_scores:
                sparse_values = [float(score) for score in bm25_scores.values()]
                sparse_min, sparse_max = min(sparse_values), max(sparse_values)
                bm25_norms = {
                    doc_id: (
                        (float(score) - sparse_min) / (sparse_max - sparse_min)
                        if sparse_max > sparse_min
                        else 1.0
                    )
                    for doc_id, score in bm25_scores.items()
                }

        bm25_ranks = {doc_id: rank for rank, doc_id in enumerate(bm25_scores)}
        raw_dense_floor = (
            min(float(item["score"]) for item in dense_candidates.values())
            if dense_candidates
            else 0.0
        )
        raw_bm25_floor = min(map(float, bm25_scores.values())) if bm25_scores else 0.0
        results: List[Dict[str, Any]] = []
        for doc_id in candidate_order:
            dense_item = dense_candidates.get(doc_id)
            dense_score = float(dense_item["score"]) if dense_item is not None else 0.0
            bm25_score = float(bm25_scores.get(doc_id, 0.0))
            dense_component = (
                dense_norms.get(doc_id, 0.0)
                if self.normalize
                else (dense_score if dense_item is not None else raw_dense_floor)
            )
            bm25_component = (
                bm25_norms.get(doc_id, 0.0)
                if self.normalize
                else (bm25_score if doc_id in bm25_scores else raw_bm25_floor)
            )
            results.append(
                {
                    "id": doc_id,
                    "text": id_to_text[doc_id],
                    "dense_score": dense_score,
                    "bm25_score": bm25_score,
                    "dense_norm": dense_norms.get(doc_id, 0.0),
                    "bm25_norm": bm25_norms.get(doc_id, 0.0),
                    "combined_score": self.alpha * dense_component
                    + (1 - self.alpha) * bm25_component,
                    "metadata": {
                        "dense_rank": dense_item["rank"] if dense_item is not None else None,
                        "bm25_rank": bm25_ranks.get(doc_id),
                        "dense_metric": dense_metric,
                    },
                }
            )

        if rerank:
            results.sort(key=lambda r: r["combined_score"], reverse=True)
        for rank, result in enumerate(results):
            result["metadata"]["rank"] = rank

        return results[:k]

    # ── BM25 backend ──────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> Dict[str, float]:
        """Run BM25 search via Pyserini or return deterministic mock scores.

        Uses a real Anserini ``SimpleSearcher`` when *pyserini_index_path*
        points to a valid Lucene index. Real mode fails closed rather than
        silently relabelling a dense-only result as hybrid retrieval.

        Args:
            query: Query string.
            k: Number of results.

        Returns:
            Mapping from doc id to BM25 score.
        """
        if self._mock_mode:
            rng = np.random.RandomState(stable_seed("hybrid_retriever.mock_bm25", query))
            return {doc_id: float(rng.uniform(0, 10)) for doc_id in self._id_map[:k]}

        # Real Pyserini BM25 search. Missing or malformed backends are research
        # integrity failures, not an implicit dense-only serving mode.
        idx_path = Path(self.pyserini_index_path)
        if not idx_path.exists():
            raise BM25BackendError(f"Pyserini index not found at {idx_path}")
        try:
            idx_path = idx_path.resolve(strict=True)
            has_segments_file = idx_path.is_dir() and any(
                child.is_file() and child.name.startswith("segments_") and child.stat().st_size > 0
                for child in idx_path.iterdir()
            )
        except OSError as exc:
            raise BM25BackendError(f"Pyserini index is not readable at {idx_path}") from exc
        if not has_segments_file:
            raise BM25BackendError(f"No Lucene index marker found at {idx_path}")

        try:
            # Pyserini loads jnius in-process.  On Windows, isolate both the JVM
            # and native teardown in a short-lived worker so a JNI failure cannot
            # corrupt the long-running evaluation process.
            if sys.platform == "win32":
                return _search_pyserini_in_worker(idx_path, query, k)

            if self._lucene_searcher is None:
                # Import the sparse Java binding directly.  The aggregate
                # ``pyserini.search.lucene`` package eagerly initializes
                # unrelated optional encoder integrations that may require
                # external credentials even though BM25 does not use them.
                from pyserini.pyclass import autoclass  # type: ignore[import-untyped]

                simple_searcher = autoclass("io.anserini.search.SimpleSearcher")
                self._lucene_searcher = simple_searcher(str(idx_path))
                logger.info(
                    "Loaded Lucene index: %d docs",
                    self._lucene_searcher.get_total_num_docs(),
                )

            hits = self._lucene_searcher.search(query, k)
            return {hit.docid: float(hit.score) for hit in hits}
        except PyseriniWorkerError:
            logger.exception("Isolated Pyserini BM25 search failed closed")
            raise
        except ImportError:
            raise BM25BackendError(
                "Pyserini is required for configured hybrid retrieval; install the locked dependency"
            ) from None
        except Exception as exc:
            raise BM25BackendError("Pyserini BM25 search failed") from exc
