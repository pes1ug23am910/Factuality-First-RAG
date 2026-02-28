"""
factuality_rag.index.builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build and persist FAISS (HNSW / IVFPQ) indexes and prepare
Pyserini-compatible JSON collections.

Functions:
    build_faiss_index  – encode passages → build FAISS index → save
    save_embeddings    – persist numpy embeddings to disk
    prepare_pyserini_collection – write Pyserini-compatible JSON docs

Example (mock-mode, no GPU)::

    >>> build_faiss_index("tests/data/sample_wiki.jsonl",
    ...     embed_model="sentence-transformers/all-mpnet-base-v2",
    ...     out_path="tmp_test.index", mock_mode=True)
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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example::

        >>> docs = load_jsonl("tests/data/sample_wiki.jsonl")
        >>> isinstance(docs, list) and len(docs) > 0
        True
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    docs = load_jsonl(jsonl_path)
    if dev_sample_size:
        docs = docs[:dev_sample_size]

    texts = [d["text"] for d in docs]
    ids = [d.get("id", str(i)) for i, d in enumerate(docs)]
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
    docs = load_jsonl(jsonl_path)
    if dev_sample_size:
        docs = docs[:dev_sample_size]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "docs.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            record = {"id": doc.get("id", ""), "contents": doc.get("text", "")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Pyserini collection (%d docs) → %s", len(docs), out)
    return str(out.resolve())
