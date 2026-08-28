#!/usr/bin/env python
"""
scripts/build_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3A-1: Download the Wikipedia corpus via HuggingFace datasets,
build FAISS + Pyserini (Lucene) indexes.

Usage::

    python scripts/build_corpus.py \\
        --n-docs 100000 \\
        --faiss-out indexes/wiki100k.faiss \\
        --pyserini-out indexes/wiki100k_lucene \\
        --embed-model sentence-transformers/all-mpnet-base-v2

Requires GPU for embedding; index building is CPU-bound.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Wikipedia corpus indexes.")
    p.add_argument("--n-docs", type=int, default=100_000, help="Number of documents to index.")
    p.add_argument(
        "--faiss-out", type=str, default="indexes/wiki100k.faiss", help="FAISS output path."
    )
    p.add_argument(
        "--pyserini-out",
        type=str,
        default="indexes/wiki100k_lucene",
        help="Pyserini Lucene output dir.",
    )
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--batch-size", type=int, default=64, help="Encoding batch size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help="Load docs from existing JSONL instead of downloading.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load or download Wikipedia passages
    if args.from_jsonl:
        logger.info("Loading docs from existing JSONL: %s", args.from_jsonl)
        docs = []
        with open(args.from_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        logger.info("Loaded %d documents from JSONL.", len(docs))
    else:
        logger.info("Loading Wikipedia passages from HuggingFace (n=%d) ...", args.n_docs)
        from factuality_rag.data.wikipedia import WikiChunker

        chunker = WikiChunker(chunk_size=200, chunk_overlap=50)
        docs = chunker.load_from_hf(sample_size=args.n_docs)
        logger.info("Loaded %d documents.", len(docs))

    from factuality_rag.index.builder import canonicalize_documents

    docs, id_list = canonicalize_documents(docs)
    if not docs:
        raise ValueError("corpus must contain at least one document")

    # 2. Build dense FAISS index
    logger.info("Building FAISS index → %s", args.faiss_out)
    import numpy as np

    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    model = SentenceTransformer(args.embed_model)
    texts = [d["text"] for d in docs]

    logger.info("Encoding %d documents (batch_size=%d) ...", len(texts), args.batch_size)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss_path = Path(args.faiss_out)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index saved (%d vectors, dim=%d).", index.ntotal, dim)

    # Save document mapping (id → text)
    doc_map_path = faiss_path.with_suffix(".json")
    doc_map = {str(i): {"id": d["id"], "text": d["text"]} for i, d in enumerate(docs)}
    with open(doc_map_path, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False)
    logger.info("Document mapping saved → %s", doc_map_path)

    # Save ID list for retriever compatibility
    ids_path = faiss_path.with_suffix(".ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(id_list, f, ensure_ascii=False)
    logger.info("ID list saved → %s (%d ids)", ids_path, len(id_list))

    corpus_path = faiss_path.with_suffix(".jsonl")
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Bound corpus sidecar saved → %s", corpus_path)

    # 3. Build the Lucene index for BM25 in an isolated process.
    logger.info("Building Pyserini Lucene index → %s", args.pyserini_out)
    from factuality_rag.index.builder import build_pyserini_index

    build_pyserini_index(str(corpus_path), args.pyserini_out)

    logger.info("Done. Corpus build complete.")


if __name__ == "__main__":
    main()
