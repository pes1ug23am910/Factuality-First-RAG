"""
factuality_rag.index.builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build and persist FAISS (HNSW / IVFPQ) and Pyserini Lucene indexes.

Functions:
    build_faiss_index  – encode passages → build FAISS index → save
    save_embeddings    – persist numpy embeddings to disk
    prepare_pyserini_collection – write Pyserini-compatible JSON docs
    build_pyserini_index – build a Lucene index in an isolated subprocess

Example (mock-mode, no GPU)::

    >>> build_faiss_index("tests/data/sample_wiki.jsonl",
    ...     embed_model="sentence-transformers/all-mpnet-base-v2",
    ...     out_path="tmp_test.index", mock_mode=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────
_faiss = None
_SentenceTransformer = None


def _get_faiss():  # type: ignore[no-untyped-def]
    """Lazy-load faiss to avoid import errors when not installed.

    Returns:
        The ``faiss`` module.
    """
    global _faiss
    if _faiss is None:
        import faiss  # type: ignore[import-untyped]

        _faiss = faiss
    return _faiss


def _get_sentence_transformer():  # type: ignore[no-untyped-def]
    """Lazy-load SentenceTransformer.

    Returns:
        The ``SentenceTransformer`` class.
    """
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer

        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


# ── Public API ────────────────────────────────────────────────


def _validate_limit(limit: Optional[int]) -> Optional[int]:
    """Validate an optional positive JSONL record limit."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("record limit must be an integer or None")
    if limit <= 0:
        raise ValueError("record limit must be positive")
    return limit


def load_jsonl(path: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.
        limit: Optional maximum number of non-empty records to read.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example::

        >>> docs = load_jsonl("tests/data/sample_wiki.jsonl")
        >>> isinstance(docs, list) and len(docs) > 0
        True
    """
    validated_limit = _validate_limit(limit)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with open(p, encoding="utf-8") as f:
        documents: List[Dict[str, Any]] = []
        for line in f:
            if not line.strip():
                continue
            documents.append(json.loads(line))
            if validated_limit is not None and len(documents) >= validated_limit:
                break
        return documents


def canonicalize_documents(
    documents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate documents and return copies with stable string identifiers."""
    canonical_documents: List[Dict[str, Any]] = []
    identifiers: List[str] = []
    seen_ids = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"corpus record {index} must be a JSON object")
        raw_id = document.get("id", str(index))
        if isinstance(raw_id, bool):
            raise ValueError(f"corpus record {index} has an invalid boolean id")
        if isinstance(raw_id, int):
            document_id = str(raw_id)
        elif isinstance(raw_id, str) and raw_id and raw_id == raw_id.strip():
            document_id = raw_id
        else:
            raise ValueError(f"corpus record {index} has an invalid document id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate corpus document id: {document_id!r}")
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"corpus record {index} has invalid text")
        seen_ids.add(document_id)
        identifiers.append(document_id)
        canonical_document = dict(document)
        canonical_document["id"] = document_id
        canonical_documents.append(canonical_document)
    return canonical_documents, identifiers


def build_faiss_index(
    jsonl_path: str,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    out_path: str = "faiss.index",
    mock_mode: bool = False,
    faiss_type: str = "hnsw_flat",
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    dim: int = 768,
    dev_sample_size: Optional[int] = None,
) -> str:
    """Build a FAISS index from a JSONL corpus.

    Args:
        jsonl_path: Path to the chunked JSONL corpus.
        embed_model: HuggingFace embedding model identifier.
        out_path: Destination path for the serialised FAISS index.
        mock_mode: If ``True``, skip model download and use random
                   embeddings (fixed seed for reproducibility).
        faiss_type: Index type – ``"hnsw_flat"`` (dev) or ``"ivfpq"``
                    (production).
        hnsw_m: HNSW graph connectivity parameter.
        hnsw_ef_construction: HNSW search depth at build time.
        dim: Embedding dimension (must match ``embed_model``).
        dev_sample_size: If set, only index the first *N* passages.

    Returns:
        Absolute path to the saved FAISS index.

    Example::

        >>> path = build_faiss_index("tests/data/sample_wiki.jsonl",
        ...     out_path="tmp.index", mock_mode=True)
        >>> Path(path).exists()
        True
    """
    docs = load_jsonl(jsonl_path, limit=dev_sample_size)
    docs, ids = canonicalize_documents(docs)
    if not docs:
        raise ValueError("corpus must contain at least one document")
    texts = [d["text"] for d in docs]
    if faiss_type not in {"hnsw_flat", "ivfpq"}:
        raise ValueError(f"Unknown faiss_type: {faiss_type}")
    if faiss_type == "ivfpq" and len(texts) < 256:
        raise ValueError(
            "ivfpq with 8-bit product quantization requires at least 256 training vectors"
        )
    logger.info("Encoding %d passages (mock=%s) ...", len(texts), mock_mode)

    if mock_mode:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(len(texts), dim).astype(np.float32)
    else:
        ST = _get_sentence_transformer()
        model = ST(embed_model)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]

    faiss = _get_faiss()

    if faiss_type == "hnsw_flat":
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        index.hnsw.efConstruction = hnsw_ef_construction
    elif faiss_type == "ivfpq":
        # TODO: tune nlist, m_pq, nbits for large-scale runs
        nlist = min(256, len(embeddings))
        m_pq = min(16, dim)
        if dim % m_pq != 0:
            raise ValueError(
                f"ivfpq embedding dimension {dim} must be divisible by subquantizers {m_pq}"
            )
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8)
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown faiss_type: {faiss_type}")

    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, type=%s", index.ntotal, faiss_type)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))

    # Persist id mapping alongside the index
    id_map_path = out.with_suffix(".ids.json")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    corpus_path = out.with_suffix(".jsonl")
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Saved FAISS index → %s", out)

    return str(out.resolve())


def save_embeddings(
    path: str,
    embeddings: Optional[np.ndarray] = None,
    jsonl_path: Optional[str] = None,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    mock_mode: bool = False,
    dim: int = 768,
) -> str:
    """Persist passage embeddings as a ``.npy`` file.

    If *embeddings* is ``None``, they are computed from *jsonl_path*.

    Args:
        path: Output ``.npy`` file path.
        embeddings: Pre-computed numpy array.
        jsonl_path: JSONL corpus path (used when *embeddings* is ``None``).
        embed_model: HuggingFace embedding model identifier.
        mock_mode: Use random embeddings.
        dim: Embedding dimension.

    Returns:
        Absolute path to the saved ``.npy`` file.

    Example::

        >>> import tempfile, os
        >>> p = os.path.join(tempfile.mkdtemp(), "emb.npy")
        >>> save_embeddings(p, embeddings=np.zeros((5, 768), dtype=np.float32))
        '...'
    """
    if embeddings is None:
        if jsonl_path is None:
            raise ValueError("Provide either 'embeddings' or 'jsonl_path'.")
        docs = load_jsonl(jsonl_path)
        texts = [d["text"] for d in docs]
        if mock_mode:
            rng = np.random.RandomState(42)
            embeddings = rng.randn(len(texts), dim).astype(np.float32)
        else:
            ST = _get_sentence_transformer()
            model = ST(embed_model)
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), embeddings)
    logger.info("Saved embeddings → %s  shape=%s", out, embeddings.shape)
    return str(out.resolve())


def prepare_pyserini_collection(
    jsonl_path: str,
    out_dir: str,
    dev_sample_size: Optional[int] = None,
) -> str:
    """Write Pyserini-compatible JSONL collection for BM25 indexing.

    Each output record has ``{"id", "contents"}``.

    Args:
        jsonl_path: Input chunked JSONL corpus.
        out_dir: Output directory for the Pyserini collection.
        dev_sample_size: Limit to the first *N* documents.

    Returns:
        Absolute path to the Pyserini collection directory.

    Example::

        >>> import tempfile
        >>> d = tempfile.mkdtemp()
        >>> prepare_pyserini_collection("tests/data/sample_wiki.jsonl", d)
        '...'
    """
    docs = load_jsonl(jsonl_path, limit=dev_sample_size)
    docs, _ = canonicalize_documents(docs)
    if not docs:
        raise ValueError("corpus must contain at least one document")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "docs.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            record = {"id": doc.get("id", ""), "contents": doc.get("text", "")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Pyserini collection (%d docs) → %s", len(docs), out)
    return str(out.resolve())


def build_pyserini_index(
    jsonl_path: str,
    out_dir: str,
    dev_sample_size: Optional[int] = None,
    *,
    threads: int = 4,
) -> str:
    """Build a real Lucene index in a short-lived Pyserini subprocess.

    The index is assembled in a same-volume temporary directory and moved into
    place only after a non-empty ``segments_*`` marker exists. Existing targets
    are never overwritten.
    """
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads must be an integer")
    if threads <= 0:
        raise ValueError("threads must be positive")

    target = Path(out_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing Lucene index: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Validate Java 21 before materializing a potentially large collection.
    # Reuse the isolated environment builder on every platform so Jarvis/Linux
    # receives the same deterministic JAVA_HOME contract as Windows.
    from factuality_rag.retriever.hybrid import _worker_environment

    environment = _worker_environment()

    with tempfile.TemporaryDirectory(prefix="factuality-rag-lucene-", dir=target.parent) as tmp:
        temporary_root = Path(tmp)
        collection = temporary_root / "collection"
        temporary_index = temporary_root / "index"
        prepare_pyserini_collection(
            jsonl_path,
            str(collection),
            dev_sample_size=dev_sample_size,
        )
        command = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection),
            "--index",
            str(temporary_index),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            env=environment,
            timeout=60 * 60,
        )
        if completed.returncode != 0:
            stderr_tail = completed.stderr[-4000:].decode("utf-8", errors="replace").strip()
            diagnostic = f"; stderr tail: {stderr_tail}" if stderr_tail else ""
            raise RuntimeError(
                f"Pyserini indexing failed with exit code {completed.returncode}; "
                f"the temporary partial index was discarded{diagnostic}"
            )
        markers = list(temporary_index.glob("segments_*"))
        if not markers or any(
            not marker.is_file() or marker.stat().st_size <= 0 for marker in markers
        ):
            raise RuntimeError("Pyserini indexing produced no valid Lucene segments marker")
        temporary_index.replace(target)

    logger.info("Built Pyserini Lucene index → %s", target)
    return str(target)
